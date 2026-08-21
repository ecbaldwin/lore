// SPDX-FileCopyrightText: 2026 Epic Games, Inc.
// SPDX-License-Identifier: MIT
//
// Cascade fork-local: verifies `StageOptions::trusted_content` /
// `commit_file`'s trust branch (see `docs/proposed/trusted-content-commit.md`
// in the cascade `server` repo) against `lore_revision`'s own public API,
// isolated from `cascade-fs-lore`.
#[cfg(test)]
mod tests {
    #![allow(clippy::disallowed_methods)] // Test fixture writes; not subject to repository write-token discipline.

    use std::io::Write;
    use std::sync::Arc;

    use bytes::Bytes;
    use lore_base::runtime::LORE_CONTEXT;
    use lore_base::runtime::runtime;
    use lore_base::types::Address;
    use lore_base::types::Context;
    use lore_revision::branch;
    use lore_revision::commit;
    use lore_revision::commit::CommitOptions;
    use lore_revision::file;
    use lore_revision::immutable;
    use lore_revision::interface::LoreArray;
    use lore_revision::interface::LoreString;
    use lore_revision::lore::BranchId;
    use lore_revision::lore::RepositoryId;
    use lore_revision::node::NodeFlags;
    use lore_revision::repository;
    use lore_revision::repository::RepositoryContext;
    use lore_revision::stage;
    use lore_revision::stage::StageOptions;
    use lore_revision::state;
    use lore_storage::options::WriteOptions;

    include!("helper.rs");

    fn write_test_file(path: &std::path::Path, content: &[u8]) {
        let mut file = std::fs::File::options()
            .create(true)
            .truncate(true)
            .read(true)
            .write(true)
            .open(path)
            .expect("Failed to create test file");
        file.write_all(content).expect("Failed to write test file");
    }

    #[tokio::test]
    #[allow(clippy::large_futures)]
    async fn trusted_content_commit_trusts_address_and_skips_real_read() {
        let (immutable_store, mutable_store, execution) =
            test_store_create().await.expect("Failed to create stores");
        let repository_id = RepositoryId::from(uuid::Uuid::now_v7());

        #[allow(clippy::disallowed_methods)]
        runtime()
            .spawn(LORE_CONTEXT.scope(execution.clone(), async move {
                let tempdir = generate_tempdir();
                let path = tempdir.to_path_buf();
                std::fs::create_dir_all(path.as_path()).expect("Create directory failed");

                let default_branch_id = BranchId::from(uuid::Uuid::now_v7());
                let write_token = repository::RepositoryWriteToken::acquire(path.as_path()).await;
                let created_repo = repository::create_local(
                    path.as_path(),
                    &write_token,
                    repository_id,
                    default_branch_id,
                    branch::DEFAULT_DEFAULT_NAME.to_string(),
                    repository::RepositoryConfig::default(),
                    false,
                )
                .await
                .expect("Failed to initialize repository");

                let repository = Arc::new(
                    RepositoryContext::new(
                        default_repository_creation_args(immutable_store, mutable_store)
                            .with_path(&path)
                            .with_id(repository_id)
                            .with_instance_id(created_repo.instance_id),
                    )
                    .with_write_token(write_token.share()),
                );
                lore_revision::instance::store_current_anchor_branch(
                    &repository,
                    default_branch_id,
                )
                .await
                .expect("Failed to store anchor branch");

                // Write the real content to the store early, as a future
                // cascade-fs-lore write path would, and get back its address.
                let trusted_bytes = Bytes::from_static(b"trusted content, written early");
                let context = Context::from(uuid::Uuid::now_v7());
                let address = immutable::write(
                    repository.clone(),
                    context,
                    trusted_bytes.clone(),
                    WriteOptions::default(),
                )
                .await
                .expect("Failed to write trusted content to store");
                let size = trusted_bytes.len() as u64;

                // The working-tree file deliberately holds different bytes —
                // if commit ever falls back to a real read, the committed
                // hash will reflect this, not the trusted content.
                let file_path = path.as_path().join("test.file");
                write_test_file(file_path.as_path(), b"WRONG on-disk content");

                let stage_signature = file::stage::stage(
                    repository.clone(),
                    &write_token,
                    LoreArray::from_vec(vec![LoreString::from(&file_path)]),
                    StageOptions {
                        case_change: stage::StageCaseChange::Error,
                        node_flags: NodeFlags::NoFlags,
                        file_id: None,
                        trusted_content: Some((address, size)),
                        no_children: false,
                        scan: true,
                    },
                )
                .await
                .expect("Failed to stage trusted content");

                let staged_state = state::State::deserialize(repository.clone(), stage_signature)
                    .await
                    .expect("Failed to deserialize staged state");
                let node_link = staged_state
                    .find_node_link(repository.clone(), "test.file")
                    .await
                    .expect("Failed to find staged node");
                let staged_node = staged_state
                    .node(repository.clone(), node_link.node)
                    .await
                    .expect("Failed to load staged node");
                assert_eq!(staged_node.address, address);
                assert_eq!(staged_node.size, size);
                assert_ne!(
                    staged_node.reserved, 0,
                    "trust marker should be set on stage"
                );

                let options = CommitOptions {
                    message: String::new(),
                    link_messages: std::collections::HashMap::new(),
                    link: None,
                    layer_messages: std::collections::HashMap::new(),
                    layer: None,
                };
                let commit_signature =
                    Box::pin(commit::commit(repository.clone(), &write_token, options))
                        .await
                        .expect("Failed to commit revision");

                let state = state::State::deserialize(repository.clone(), commit_signature)
                    .await
                    .expect("Failed to deserialize committed state");
                let node_link = state
                    .find_node_link(repository.clone(), "test.file")
                    .await
                    .expect("Failed to find committed node");
                let committed_node = state
                    .node(repository.clone(), node_link.node)
                    .await
                    .expect("Failed to load committed node");

                // The committed address/size must be the trusted ones, not a
                // hash of the (wrong) on-disk bytes.
                assert_eq!(committed_node.address, address);
                assert_eq!(committed_node.size, size);
                assert_eq!(
                    committed_node.reserved, 0,
                    "marker must be cleared after commit"
                );

                let options = immutable::read_options_from_repository(&repository);
                let read_back =
                    immutable::read(repository.clone(), committed_node.address, None, options)
                        .await
                        .expect("Failed to read back committed content");
                assert_eq!(read_back.as_ref(), trusted_bytes.as_ref());

                let _ = std::fs::remove_dir_all(path.as_path());
            }))
            .await
            .expect("Test task failed");
    }

    #[tokio::test]
    #[allow(clippy::large_futures)]
    async fn trusted_content_falls_back_when_not_stored_locally() {
        let (immutable_store, mutable_store, execution) =
            test_store_create().await.expect("Failed to create stores");
        let repository_id = RepositoryId::from(uuid::Uuid::now_v7());

        #[allow(clippy::disallowed_methods)]
        runtime()
            .spawn(LORE_CONTEXT.scope(execution.clone(), async move {
                let tempdir = generate_tempdir();
                let path = tempdir.to_path_buf();
                std::fs::create_dir_all(path.as_path()).expect("Create directory failed");

                let default_branch_id = BranchId::from(uuid::Uuid::now_v7());
                let write_token = repository::RepositoryWriteToken::acquire(path.as_path()).await;
                let created_repo = repository::create_local(
                    path.as_path(),
                    &write_token,
                    repository_id,
                    default_branch_id,
                    branch::DEFAULT_DEFAULT_NAME.to_string(),
                    repository::RepositoryConfig::default(),
                    false,
                )
                .await
                .expect("Failed to initialize repository");

                let repository = Arc::new(
                    RepositoryContext::new(
                        default_repository_creation_args(immutable_store, mutable_store)
                            .with_path(&path)
                            .with_id(repository_id)
                            .with_instance_id(created_repo.instance_id),
                    )
                    .with_write_token(write_token.share()),
                );
                lore_revision::instance::store_current_anchor_branch(
                    &repository,
                    default_branch_id,
                )
                .await
                .expect("Failed to store anchor branch");

                // A trusted address that was never actually written to this
                // store — stands in for content written early and evicted
                // (or any other reason it's no longer present) before commit.
                let bogus_address = rand::random::<Address>();

                let real_bytes = b"the real on-disk content commit must fall back to";
                let file_path = path.as_path().join("test.file");
                write_test_file(file_path.as_path(), real_bytes);

                file::stage::stage(
                    repository.clone(),
                    &write_token,
                    LoreArray::from_vec(vec![LoreString::from(&file_path)]),
                    StageOptions {
                        case_change: stage::StageCaseChange::Error,
                        node_flags: NodeFlags::NoFlags,
                        file_id: None,
                        trusted_content: Some((bogus_address, 99999)),
                        no_children: false,
                        scan: true,
                    },
                )
                .await
                .expect("Failed to stage with a bogus trusted address");

                let options = CommitOptions {
                    message: String::new(),
                    link_messages: std::collections::HashMap::new(),
                    link: None,
                    layer_messages: std::collections::HashMap::new(),
                    layer: None,
                };
                let commit_signature =
                    Box::pin(commit::commit(repository.clone(), &write_token, options))
                        .await
                        .expect("Commit should fall back to a real read, not fail");

                let state = state::State::deserialize(repository.clone(), commit_signature)
                    .await
                    .expect("Failed to deserialize committed state");
                let node_link = state
                    .find_node_link(repository.clone(), "test.file")
                    .await
                    .expect("Failed to find committed node");
                let committed_node = state
                    .node(repository.clone(), node_link.node)
                    .await
                    .expect("Failed to load committed node");

                // Must NOT have trusted the dangling address — the real
                // file was read, chunked and hashed instead.
                assert_ne!(committed_node.address, bogus_address);
                assert_eq!(committed_node.size, real_bytes.len() as u64);

                let read_options = immutable::read_options_from_repository(&repository);
                let read_back = immutable::read(
                    repository.clone(),
                    committed_node.address,
                    None,
                    read_options,
                )
                .await
                .expect("Failed to read back committed content");
                assert_eq!(read_back.as_ref(), real_bytes);

                let _ = std::fs::remove_dir_all(path.as_path());
            }))
            .await
            .expect("Test task failed");
    }

    #[tokio::test]
    #[allow(clippy::large_futures)]
    async fn trusted_content_marker_does_not_leak_into_next_normal_stage() {
        let (immutable_store, mutable_store, _unused_execution) =
            test_store_create().await.expect("Failed to create stores");
        // Re-staging an already-staged (uncommitted) node is normally a
        // no-op unless forced -- force it here so the second, real-content
        // stage call actually reconciles against the filesystem instead of
        // leaving the first trust-stage's node untouched.
        let execution = Arc::new(
            lore_revision::interface::ExecutionContext::new_client_with_user_id(
                lore_revision::interface::LoreGlobalArgs {
                    force: 1,
                    ..Default::default()
                },
                lore_revision::relay::EventDispatcher::no_dispatch(),
                "test-user".to_string(),
            ),
        );
        let repository_id = RepositoryId::from(uuid::Uuid::now_v7());

        #[allow(clippy::disallowed_methods)]
        runtime()
            .spawn(LORE_CONTEXT.scope(execution.clone(), async move {
                let tempdir = generate_tempdir();
                let path = tempdir.to_path_buf();
                std::fs::create_dir_all(path.as_path()).expect("Create directory failed");

                let default_branch_id = BranchId::from(uuid::Uuid::now_v7());
                let write_token = repository::RepositoryWriteToken::acquire(path.as_path()).await;
                let created_repo = repository::create_local(
                    path.as_path(),
                    &write_token,
                    repository_id,
                    default_branch_id,
                    branch::DEFAULT_DEFAULT_NAME.to_string(),
                    repository::RepositoryConfig::default(),
                    false,
                )
                .await
                .expect("Failed to initialize repository");

                let repository = Arc::new(
                    RepositoryContext::new(
                        default_repository_creation_args(immutable_store, mutable_store)
                            .with_path(&path)
                            .with_id(repository_id)
                            .with_instance_id(created_repo.instance_id),
                    )
                    .with_write_token(write_token.share()),
                );
                lore_revision::instance::store_current_anchor_branch(
                    &repository,
                    default_branch_id,
                )
                .await
                .expect("Failed to store anchor branch");

                // First: trust-stage a brand new file.
                let trusted_bytes = Bytes::from_static(b"trusted content A");
                let context = Context::from(uuid::Uuid::now_v7());
                let address_a = immutable::write(
                    repository.clone(),
                    context,
                    trusted_bytes.clone(),
                    WriteOptions::default(),
                )
                .await
                .expect("Failed to write trusted content A to store");
                let size_a = trusted_bytes.len() as u64;

                let file_path = path.as_path().join("leak.file");
                write_test_file(file_path.as_path(), trusted_bytes.as_ref());

                let stage_signature = file::stage::stage(
                    repository.clone(),
                    &write_token,
                    LoreArray::from_vec(vec![LoreString::from(&file_path)]),
                    StageOptions {
                        case_change: stage::StageCaseChange::Error,
                        node_flags: NodeFlags::NoFlags,
                        file_id: None,
                        trusted_content: Some((address_a, size_a)),
                        no_children: false,
                        scan: true,
                    },
                )
                .await
                .expect("Failed to trust-stage content A");

                let staged_state = state::State::deserialize(repository.clone(), stage_signature)
                    .await
                    .expect("Failed to deserialize staged state");
                let node_link = staged_state
                    .find_node_link(repository.clone(), "leak.file")
                    .await
                    .expect("Failed to find staged node");
                let staged_node = staged_state
                    .node(repository.clone(), node_link.node)
                    .await
                    .expect("Failed to load staged node");
                assert_ne!(
                    staged_node.reserved, 0,
                    "trust marker should be set after the trust-stage"
                );

                // Second: real content change, staged normally (no
                // trusted_content) over the SAME path.
                let real_bytes = b"real content B, staged normally over the trusted node";
                write_test_file(file_path.as_path(), real_bytes);

                let stage_signature = file::stage::stage(
                    repository.clone(),
                    &write_token,
                    LoreArray::from_vec(vec![LoreString::from(&file_path)]),
                    StageOptions {
                        case_change: stage::StageCaseChange::Error,
                        node_flags: NodeFlags::NoFlags,
                        file_id: None,
                        trusted_content: None,
                        no_children: false,
                        scan: true,
                    },
                )
                .await
                .expect("Failed to stage content B normally");

                let staged_state = state::State::deserialize(repository.clone(), stage_signature)
                    .await
                    .expect("Failed to deserialize re-staged state");
                let node_link = staged_state
                    .find_node_link(repository.clone(), "leak.file")
                    .await
                    .expect("Failed to find re-staged node");
                let staged_node = staged_state
                    .node(repository.clone(), node_link.node)
                    .await
                    .expect("Failed to load re-staged node");
                assert_eq!(
                    staged_node.reserved, 0,
                    "trust marker must not survive a subsequent normal stage"
                );

                let options = CommitOptions {
                    message: String::new(),
                    link_messages: std::collections::HashMap::new(),
                    link: None,
                    layer_messages: std::collections::HashMap::new(),
                    layer: None,
                };
                let commit_signature =
                    Box::pin(commit::commit(repository.clone(), &write_token, options))
                        .await
                        .expect("Failed to commit revision");

                let state = state::State::deserialize(repository.clone(), commit_signature)
                    .await
                    .expect("Failed to deserialize committed state");
                let node_link = state
                    .find_node_link(repository.clone(), "leak.file")
                    .await
                    .expect("Failed to find committed node");
                let committed_node = state
                    .node(repository.clone(), node_link.node)
                    .await
                    .expect("Failed to load committed node");

                // Must have hashed the new real content B, not trusted the
                // stale address from the first trust-stage of content A.
                assert_ne!(committed_node.address, address_a);
                assert_eq!(committed_node.size, real_bytes.len() as u64);

                let read_options = immutable::read_options_from_repository(&repository);
                let read_back = immutable::read(
                    repository.clone(),
                    committed_node.address,
                    None,
                    read_options,
                )
                .await
                .expect("Failed to read back committed content");
                assert_eq!(read_back.as_ref(), real_bytes);

                let _ = std::fs::remove_dir_all(path.as_path());
            }))
            .await
            .expect("Test task failed");
    }
}
