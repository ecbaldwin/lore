// SPDX-FileCopyrightText: 2026 Epic Games, Inc.
// SPDX-License-Identifier: MIT
//
// Cascade fork-local: a node's file metadata read and replaced against an
// in-memory `State` (`metadata::in_state`), and the undelete path's reset of
// it (`direct_write`'s `clear_staged_delete`).
#[cfg(test)]
mod tests {
    #![allow(clippy::disallowed_methods)] // Test fixture writes; not subject to repository write-token discipline.

    use std::sync::Arc;

    use bytes::Bytes;
    use lore_base::runtime::LORE_CONTEXT;
    use lore_base::runtime::runtime;
    use lore_base::types::Context;
    use lore_base::types::Hash;
    use lore_revision::branch;
    use lore_revision::commit;
    use lore_revision::direct_write;
    use lore_revision::direct_write::LeafContent;
    use lore_revision::immutable;
    use lore_revision::lore::BranchId;
    use lore_revision::lore::RepositoryId;
    use lore_revision::metadata::Metadata;
    use lore_revision::metadata::in_state;
    use lore_revision::node::NodeID;
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

    async fn write_leaf(repository: &Arc<RepositoryContext>, bytes: &[u8]) -> LeafContent {
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

    /// Runs `body` inside the lore execution context every direct-write
    /// entry point needs, against a fresh temp repository. Mirrors
    /// `direct_write.rs`'s own test setup.
    async fn with_repository<F, Fut>(body: F)
    where
        F: FnOnce(
                Arc<RepositoryContext>,
                repository::RepositoryWriteToken,
                Arc<state::State>,
            ) -> Fut
            + Send
            + 'static,
        Fut: std::future::Future<Output = ()> + Send,
    {
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
                let tree_state = state::State::new();

                body(repository, write_token, tree_state).await;

                let _ = std::fs::remove_dir_all(tempdir.path());
            }))
            .await
            .expect("Test task failed");
    }

    async fn resync(
        repository: &Arc<RepositoryContext>,
        write_token: &repository::RepositoryWriteToken,
        tree_state: &Arc<state::State>,
    ) -> Arc<state::State> {
        commit::rehash_staged_directory(repository.clone(), tree_state.clone(), ROOT_NODE)
            .await
            .expect("Failed to eagerly rehash directory ancestors");
        let signature = tree_state
            .serialize(repository.clone(), write_token)
            .await
            .expect("Failed to serialize");
        state::State::deserialize(repository.clone(), signature)
            .await
            .expect("Failed to deserialize fresh state")
    }

    async fn stage(
        repository: &Arc<RepositoryContext>,
        tree_state: &Arc<state::State>,
        path: &str,
        bytes: &[u8],
    ) -> NodeID {
        let content = write_leaf(repository, bytes).await;
        direct_write::stage_leaf_from_facts(repository.clone(), tree_state.clone(), path, content)
            .await
            .expect("Failed to stage")
    }

    fn with_key(key: &str, value: &[u8]) -> Metadata {
        let mut metadata = Metadata::new();
        metadata.set_binary(key, value).expect("set_binary");
        metadata
    }

    async fn value(
        repository: &Arc<RepositoryContext>,
        tree_state: &state::State,
        node: NodeID,
        key: &str,
    ) -> Option<Vec<u8>> {
        let (_, metadata) = in_state::file_metadata_load(repository.clone(), tree_state, node)
            .await
            .expect("load");
        metadata.get_binary(key).ok().map(<[u8]>::to_vec)
    }

    #[tokio::test]
    #[allow(clippy::large_futures)]
    async fn stored_metadata_reads_back_and_survives_serialization() {
        with_repository(|repository, write_token, tree_state| async move {
            let node = stage(&repository, &tree_state, "d/a.txt", b"content").await;
            let (none, empty) =
                in_state::file_metadata_load(repository.clone(), &tree_state, node)
                    .await
                    .expect("load");
            assert!(none.is_zero());
            assert!(empty.is_empty());

            let stored = in_state::file_metadata_store(
                repository.clone(),
                &tree_state,
                node,
                none,
                &with_key("xattr.k", b"v"),
            )
            .await
            .expect("store")
            .expect("the slot was as loaded");
            assert!(!stored.is_zero());
            assert_eq!(
                value(&repository, &tree_state, node, "xattr.k").await,
                Some(b"v".to_vec())
            );

            let fresh = resync(&repository, &write_token, &tree_state).await;
            let (hash, _) = in_state::file_metadata_load(repository.clone(), &fresh, node)
                .await
                .expect("load");
            assert_eq!(hash, stored);
            assert_eq!(
                value(&repository, &fresh, node, "xattr.k").await,
                Some(b"v".to_vec())
            );
            let directory = fresh
                .find_node_link(repository.clone(), "d")
                .await
                .expect("find d")
                .node;
            let (hash, _) = in_state::file_metadata_load(repository.clone(), &fresh, directory)
                .await
                .expect("load d");
            assert!(hash.is_zero(), "the parent carries none of its own");
        })
        .await;
    }

    #[tokio::test]
    #[allow(clippy::large_futures)]
    async fn a_store_built_on_a_stale_hash_changes_nothing() {
        with_repository(|repository, _write_token, tree_state| async move {
            let node = stage(&repository, &tree_state, "a.txt", b"content").await;
            let first = in_state::file_metadata_store(
                repository.clone(),
                &tree_state,
                node,
                Hash::default(),
                &with_key("xattr.k", b"first"),
            )
            .await
            .expect("store")
            .expect("stored");

            let stale = in_state::file_metadata_store(
                repository.clone(),
                &tree_state,
                node,
                Hash::default(),
                &with_key("xattr.k", b"second"),
            )
            .await
            .expect("store");
            assert_eq!(stale, None);
            assert_eq!(
                value(&repository, &tree_state, node, "xattr.k").await,
                Some(b"first".to_vec())
            );

            let cleared = in_state::file_metadata_store(
                repository.clone(),
                &tree_state,
                node,
                first,
                &Metadata::new(),
            )
            .await
            .expect("store")
            .expect("built on the current hash");
            assert!(cleared.is_zero());
            assert_eq!(value(&repository, &tree_state, node, "xattr.k").await, None);
        })
        .await;
    }

    #[tokio::test]
    #[allow(clippy::large_futures)]
    async fn a_directory_carries_metadata_too() {
        with_repository(|repository, write_token, tree_state| async move {
            stage(&repository, &tree_state, "d/a.txt", b"content").await;
            let directory = tree_state
                .find_node_link(repository.clone(), "d")
                .await
                .expect("find d")
                .node;
            in_state::file_metadata_store(
                repository.clone(),
                &tree_state,
                directory,
                Hash::default(),
                &with_key("xattr.k", b"dir"),
            )
            .await
            .expect("store")
            .expect("stored");

            let fresh = resync(&repository, &write_token, &tree_state).await;
            assert_eq!(
                value(&repository, &fresh, directory, "xattr.k").await,
                Some(b"dir".to_vec())
            );
        })
        .await;
    }

    #[tokio::test]
    #[allow(clippy::large_futures)]
    async fn recreating_a_deleted_path_starts_with_no_metadata() {
        with_repository(|repository, write_token, tree_state| async move {
            let node = stage(&repository, &tree_state, "a.txt", b"first").await;
            in_state::file_metadata_store(
                repository.clone(),
                &tree_state,
                node,
                Hash::default(),
                &with_key("xattr.k", b"v"),
            )
            .await
            .expect("store")
            .expect("stored");

            direct_write::stage_delete_from_facts(repository.clone(), tree_state.clone(), "a.txt")
                .await
                .expect("delete must not error")
                .expect("a.txt was live");
            let recreated = stage(&repository, &tree_state, "a.txt", b"second").await;
            assert_eq!(recreated, node, "undelete reuses the node");

            let fresh = resync(&repository, &write_token, &tree_state).await;
            let (hash, _) = in_state::file_metadata_load(repository.clone(), &fresh, node)
                .await
                .expect("load");
            assert!(hash.is_zero(), "the recreated file has the old one's metadata");
        })
        .await;
    }

    #[tokio::test]
    #[allow(clippy::large_futures)]
    async fn a_node_staged_for_deletion_takes_no_metadata() {
        with_repository(|repository, _write_token, tree_state| async move {
            let node = stage(&repository, &tree_state, "a.txt", b"first").await;
            direct_write::stage_delete_from_facts(repository.clone(), tree_state.clone(), "a.txt")
                .await
                .expect("delete must not error")
                .expect("a.txt was live");
            let refused = in_state::file_metadata_store(
                repository.clone(),
                &tree_state,
                node,
                Hash::default(),
                &with_key("xattr.k", b"v"),
            )
            .await;
            assert!(refused.is_err());
            let still = tree_state.node(repository.clone(), node).await.expect("node");
            assert!(still.is_staged_delete(), "the store must not undelete it");
        })
        .await;
    }
}
