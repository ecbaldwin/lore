// SPDX-FileCopyrightText: 2026 Epic Games, Inc.
// SPDX-License-Identifier: MIT
//
// Cascade fork-local: verifies checkpoint 4's metadata-free entry points
// (`direct_write::{stage_directory_from_facts, stage_delete_from_facts,
// stage_move_from_facts}`, docs/proposed/direct-to-store-writes.md §6, in
// the cascade `server` repo) and the undelete/type-mismatch reconciliation
// `stage_one_component` now shares with them, isolated from cascade-fs-lore.
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
                let tree_state = Arc::new(state::State::new());

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

    #[tokio::test]
    #[allow(clippy::large_futures)]
    async fn stage_directory_from_facts_creates_empty_directory() {
        with_repository(|repository, write_token, tree_state| async move {
            direct_write::stage_directory_from_facts(repository.clone(), tree_state.clone(), "a/b")
                .await
                .expect("Failed to stage a/b as a directory");

            let fresh = resync(&repository, &write_token, &tree_state).await;
            let link = fresh
                .find_node_link(repository.clone(), "a/b")
                .await
                .expect("Failed to find a/b");
            let node = fresh
                .node(repository.clone(), link.node)
                .await
                .expect("Failed to load a/b");
            assert!(node.is_directory());
            assert!(!node.is_staged_delete());
        })
        .await;
    }

    #[tokio::test]
    #[allow(clippy::large_futures)]
    async fn stage_delete_from_facts_hides_the_node_and_is_idempotent() {
        with_repository(|repository, write_token, tree_state| async move {
            let content = write_leaf(&repository, b"deleteme").await;
            direct_write::stage_leaf_from_facts(
                repository.clone(),
                tree_state.clone(),
                "a.txt",
                content,
            )
            .await
            .expect("Failed to stage a.txt");

            let deleted = direct_write::stage_delete_from_facts(
                repository.clone(),
                tree_state.clone(),
                "a.txt",
            )
            .await
            .expect("delete must not error");
            assert!(deleted.is_some(), "a live node must report Some(node_id)");

            let fresh = resync(&repository, &write_token, &tree_state).await;
            // `find_node_link` still resolves a staged-delete node by path
            // (see read.rs's own separate `is_staged_delete()` check in the
            // cascade crate) -- the tombstone is the flag, not absence from
            // the tree.
            let link = fresh
                .find_node_link(repository.clone(), "a.txt")
                .await
                .expect("staged-delete node stays resolvable by path");
            let node = fresh
                .node(repository.clone(), link.node)
                .await
                .expect("Failed to load a.txt");
            assert!(node.is_staged_delete());

            // Idempotent: deleting an already-deleted (or absent) path is a
            // no-op, not an error -- matches WebDAV DELETE's contract.
            let again = direct_write::stage_delete_from_facts(
                repository.clone(),
                tree_state.clone(),
                "a.txt",
            )
            .await
            .expect("re-deleting must not error");
            assert!(again.is_none());

            let missing = direct_write::stage_delete_from_facts(
                repository.clone(),
                tree_state.clone(),
                "never/existed.txt",
            )
            .await
            .expect("deleting an absent path must not error");
            assert!(missing.is_none());
        })
        .await;
    }

    #[tokio::test]
    #[allow(clippy::large_futures)]
    async fn stage_move_from_facts_relocates_content_and_replaces_destination() {
        with_repository(|repository, write_token, tree_state| async move {
            let content = write_leaf(&repository, b"moveme").await;
            let original = direct_write::stage_leaf_from_facts(
                repository.clone(),
                tree_state.clone(),
                "src.txt",
                content,
            )
            .await
            .expect("Failed to stage src.txt");

            let existing_dest = write_leaf(&repository, b"will be overwritten").await;
            direct_write::stage_leaf_from_facts(
                repository.clone(),
                tree_state.clone(),
                "dst/dest.txt",
                existing_dest,
            )
            .await
            .expect("Failed to stage dst/dest.txt");

            let moved = direct_write::stage_move_from_facts(
                repository.clone(),
                tree_state.clone(),
                "src.txt",
                "dst/dest.txt",
            )
            .await
            .expect("Failed to move src.txt -> dst/dest.txt");
            assert_eq!(moved, original, "a move keeps the node's identity");

            let fresh = resync(&repository, &write_token, &tree_state).await;

            let src_gone = fresh.find_node_link(repository.clone(), "src.txt").await;
            if let Ok(link) = src_gone {
                let node = fresh
                    .node(repository.clone(), link.node)
                    .await
                    .expect("Failed to load stale src.txt node");
                assert!(
                    node.is_staged_delete(),
                    "old parent must lose the moved node"
                );
            }

            let link = fresh
                .find_node_link(repository.clone(), "dst/dest.txt")
                .await
                .expect("Failed to find dst/dest.txt");
            assert_eq!(link.node, original);
            let node = fresh
                .node(repository.clone(), link.node)
                .await
                .expect("Failed to load dst/dest.txt");
            assert_eq!(node.address, content.address);
            assert_eq!(node.size, content.size);
            assert!(!node.is_staged_delete());
        })
        .await;
    }

    #[tokio::test]
    #[allow(clippy::large_futures)]
    async fn recreating_a_deleted_path_undeletes_instead_of_staying_hidden() {
        with_repository(|repository, write_token, tree_state| async move {
            let first = write_leaf(&repository, b"first").await;
            direct_write::stage_leaf_from_facts(
                repository.clone(),
                tree_state.clone(),
                "a.txt",
                first,
            )
            .await
            .expect("Failed to stage a.txt");

            direct_write::stage_delete_from_facts(repository.clone(), tree_state.clone(), "a.txt")
                .await
                .expect("delete must not error")
                .expect("a.txt was live");

            let second = write_leaf(&repository, b"second, after delete").await;
            let node_id = direct_write::stage_leaf_from_facts(
                repository.clone(),
                tree_state.clone(),
                "a.txt",
                second,
            )
            .await
            .expect("re-creating a deleted path must undelete, not error");

            let fresh = resync(&repository, &write_token, &tree_state).await;
            let link = fresh
                .find_node_link(repository.clone(), "a.txt")
                .await
                .expect("Failed to find a.txt");
            assert_eq!(link.node, node_id, "undelete reuses the same node identity");
            let node = fresh
                .node(repository.clone(), link.node)
                .await
                .expect("Failed to load a.txt");
            assert!(
                !node.is_staged_delete(),
                "recreating the path must clear the staged-delete flag"
            );
            assert_eq!(node.address, second.address);
            assert_eq!(node.size, second.size);
        })
        .await;
    }

    #[tokio::test]
    #[allow(clippy::large_futures)]
    async fn recreating_a_deleted_path_as_a_different_kind_adds_alongside_it() {
        with_repository(|repository, write_token, tree_state| async move {
            let content = write_leaf(&repository, b"a file here first").await;
            direct_write::stage_leaf_from_facts(
                repository.clone(),
                tree_state.clone(),
                "a",
                content,
            )
            .await
            .expect("Failed to stage 'a' as a file");

            direct_write::stage_delete_from_facts(repository.clone(), tree_state.clone(), "a")
                .await
                .expect("delete must not error")
                .expect("'a' was live");

            // A directory now belongs at the same name -- must succeed by
            // adding a fresh node, not error out on the deleted file's type.
            direct_write::stage_directory_from_facts(repository.clone(), tree_state.clone(), "a")
                .await
                .expect("staging a directory over a deleted file's path must succeed");

            let fresh = resync(&repository, &write_token, &tree_state).await;
            let link = fresh
                .find_node_link(repository.clone(), "a")
                .await
                .expect("Failed to find 'a'");
            let node = fresh
                .node(repository.clone(), link.node)
                .await
                .expect("Failed to load 'a'");
            assert!(node.is_directory());
            assert!(!node.is_staged_delete());
        })
        .await;
    }
}
