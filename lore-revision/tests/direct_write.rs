// SPDX-FileCopyrightText: 2026 Epic Games, Inc.
// SPDX-License-Identifier: MIT
//
// Cascade fork-local: verifies `direct_write::stage_leaf_from_facts` (the
// checkpoint-3 metadata-free single-path staging entry point from
// docs/proposed/direct-to-store-writes.md, §2/§5, in the cascade `server`
// repo), isolated from cascade-fs-lore.
#[cfg(test)]
mod tests {
    #![allow(clippy::disallowed_methods)] // Test fixture writes; not subject to repository write-token discipline.

    use std::sync::Arc;

    use bytes::Bytes;
    use lore_base::runtime::LORE_CONTEXT;
    use lore_base::runtime::runtime;
    use lore_base::types::Context;
    use lore_revision::branch;
    use lore_revision::commit;
    use lore_revision::direct_write;
    use lore_revision::direct_write::LeafContent;
    use lore_revision::immutable;
    use lore_revision::lore::BranchId;
    use lore_revision::lore::RepositoryId;
    use lore_revision::node::ROOT_NODE;
    use lore_revision::repository;
    use lore_revision::repository::RepositoryContext;
    use lore_revision::state;
    use lore_storage::options::WriteOptions;

    include!("helper.rs");

    async fn new_repository(
        immutable_store: Arc<dyn lore_storage::ImmutableStore>,
        mutable_store: Arc<dyn lore_storage::MutableStore>,
        path: &std::path::Path,
    ) -> (Arc<RepositoryContext>, repository::RepositoryWriteToken) {
        let repository_id = RepositoryId::from(uuid::Uuid::now_v7());
        let default_branch_id = BranchId::from(uuid::Uuid::now_v7());
        let write_token = repository::RepositoryWriteToken::acquire(path).await;
        let created_repo = repository::create_local(
            path,
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
                    .with_path(path)
                    .with_id(repository_id)
                    .with_instance_id(created_repo.instance_id),
            )
            .with_write_token(write_token.share()),
        );
        lore_revision::instance::store_current_anchor_branch(&repository, default_branch_id)
            .await
            .expect("Failed to store anchor branch");

        (repository, write_token)
    }

    async fn write_leaf(
        repository: &Arc<RepositoryContext>,
        bytes: &[u8],
    ) -> LeafContent {
        let context = Context::from(uuid::Uuid::now_v7());
        let address = immutable::write(
            repository.clone(),
            context,
            Bytes::copy_from_slice(bytes),
            WriteOptions::default(),
        )
        .await
        .expect("Failed to write content to store");
        LeafContent {
            address,
            size: bytes.len() as u64,
            mode: 0,
        }
    }

    #[tokio::test]
    #[allow(clippy::large_futures)]
    async fn stage_leaf_from_facts_builds_path_and_survives_fresh_read() {
        let execution = setup_test_execution();

        #[allow(clippy::disallowed_methods)]
        runtime()
            .spawn(LORE_CONTEXT.scope(execution.clone(), async move {
                let (immutable_store, mutable_store, _) =
                    test_store_create().await.expect("Failed to create stores");
                let tempdir = generate_tempdir();
                std::fs::create_dir_all(tempdir.path()).expect("Create directory failed");
                let (repository, write_token) =
                    new_repository(immutable_store, mutable_store, tempdir.path()).await;

                let content = write_leaf(&repository, b"first content").await;
                let tree_state = Arc::new(state::State::new());

                let leaf = direct_write::stage_leaf_from_facts(
                    repository.clone(),
                    tree_state.clone(),
                    "a/b/c.txt",
                    content,
                )
                .await
                .expect("Failed to stage a/b/c.txt from facts");

                commit::rehash_staged_directory(repository.clone(), tree_state.clone(), ROOT_NODE)
                    .await
                    .expect("Failed to eagerly rehash directory ancestors");

                let signature = tree_state
                    .serialize(repository.clone(), &write_token)
                    .await
                    .expect("Failed to serialize");

                let fresh = state::State::deserialize(repository.clone(), signature)
                    .await
                    .expect("Failed to deserialize fresh state");
                let node_link = fresh
                    .find_node_link(repository.clone(), "a/b/c.txt")
                    .await
                    .expect("Failed to find a/b/c.txt");
                assert_eq!(node_link.node, leaf);
                let leaf_node = fresh
                    .node(repository.clone(), node_link.node)
                    .await
                    .expect("Failed to load leaf node");
                assert_eq!(leaf_node.address, content.address);
                assert_eq!(leaf_node.size, content.size);
                assert!(leaf_node.is_file());

                let dir_a = fresh
                    .find_node_link(repository.clone(), "a")
                    .await
                    .expect("Failed to find 'a'");
                let dir_a_node = fresh
                    .node(repository.clone(), dir_a.node)
                    .await
                    .expect("Failed to load 'a'");
                assert!(
                    !dir_a_node.address.hash.is_zero(),
                    "ancestor directory must have a real eager Merkle hash"
                );

                // Second write to the SAME path: exercises the "leaf already
                // exists" branch (State::node_modify-equivalent update),
                // not the "create new node" branch the first write took.
                let content2 = write_leaf(&repository, b"second, different content").await;
                let leaf2 = direct_write::stage_leaf_from_facts(
                    repository.clone(),
                    tree_state.clone(),
                    "a/b/c.txt",
                    content2,
                )
                .await
                .expect("Failed to re-stage a/b/c.txt from facts");
                assert_eq!(leaf2, leaf, "overwriting the same path reuses the same node");

                commit::rehash_staged_directory(repository.clone(), tree_state.clone(), ROOT_NODE)
                    .await
                    .expect("Failed to eagerly rehash directory ancestors (second write)");

                let signature2 = tree_state
                    .serialize(repository.clone(), &write_token)
                    .await
                    .expect("Failed to serialize (second write)");
                let fresh2 = state::State::deserialize(repository.clone(), signature2)
                    .await
                    .expect("Failed to deserialize fresh state (second write)");
                let node_link2 = fresh2
                    .find_node_link(repository.clone(), "a/b/c.txt")
                    .await
                    .expect("Failed to find a/b/c.txt (second write)");
                let leaf_node2 = fresh2
                    .node(repository.clone(), node_link2.node)
                    .await
                    .expect("Failed to load leaf node (second write)");
                assert_eq!(leaf_node2.address, content2.address);
                assert_eq!(leaf_node2.size, content2.size);
                assert_ne!(
                    leaf_node2.address, content.address,
                    "the second write's address must replace, not merge with, the first"
                );

                let dir_a2 = fresh2
                    .node(repository.clone(), dir_a.node)
                    .await
                    .expect("Failed to load 'a' (second write)");
                assert_ne!(
                    dir_a2.address, dir_a_node.address,
                    "the ancestor's Merkle hash must change when its content changes"
                );

                let _ = std::fs::remove_dir_all(tempdir.path());
            }))
            .await
            .expect("Test task failed");
    }

    #[tokio::test]
    #[allow(clippy::large_futures)]
    async fn stage_leaf_from_facts_rejects_type_mismatch() {
        let execution = setup_test_execution();

        #[allow(clippy::disallowed_methods)]
        runtime()
            .spawn(LORE_CONTEXT.scope(execution.clone(), async move {
                let (immutable_store, mutable_store, _) =
                    test_store_create().await.expect("Failed to create stores");
                let tempdir = generate_tempdir();
                std::fs::create_dir_all(tempdir.path()).expect("Create directory failed");
                let (repository, _write_token) =
                    new_repository(immutable_store, mutable_store, tempdir.path()).await;

                let content = write_leaf(&repository, b"leaf at 'a'").await;
                let tree_state = Arc::new(state::State::new());

                // Stage a *file* at "a" first.
                direct_write::stage_leaf_from_facts(
                    repository.clone(),
                    tree_state.clone(),
                    "a",
                    content,
                )
                .await
                .expect("Failed to stage 'a' as a file");

                // Now try to stage "a/b.txt", which needs "a" to be a
                // directory. Must be rejected, not silently reconciled.
                let content2 = write_leaf(&repository, b"leaf at 'a/b.txt'").await;
                let result = direct_write::stage_leaf_from_facts(
                    repository.clone(),
                    tree_state.clone(),
                    "a/b.txt",
                    content2,
                )
                .await;
                assert!(
                    result.is_err(),
                    "staging through an existing file must fail, not silently corrupt the tree"
                );

                let _ = std::fs::remove_dir_all(tempdir.path());
            }))
            .await
            .expect("Test task failed");
    }
}
