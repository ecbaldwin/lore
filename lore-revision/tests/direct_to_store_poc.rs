// SPDX-FileCopyrightText: 2026 Epic Games, Inc.
// SPDX-License-Identifier: MIT
//
// Cascade fork-local: checkpoint 1 of
// `docs/proposed/direct-to-store-writes.md` in the cascade `server` repo —
// "prove the concept inside lore-revision, isolated from cascade." Builds a
// node chain root-to-leaf from explicit, caller-supplied facts
// (name/is-directory/address/size) using only the existing `pub`
// `state::State` primitives (`node_add`, `find_node_link`, `node_children`),
// with no `lore_io` filesystem access anywhere in the call path — unlike
// `stage_filesystem_path`/`stage_node_from_metadata`
// (`lore-revision/src/stage.rs`), which stat a real path at every level.
//
// This does not touch `file::stage::stage`, `StageOptions`, or `commit()` —
// those remain the fs-driven orchestration the design doc's §5 wants to
// eventually grow metadata-free siblings of. This test only establishes that
// the underlying tree-mutation primitives they're built on already support
// driving them from facts instead of a stat.
#[cfg(test)]
mod tests {
    #![allow(clippy::disallowed_methods)] // Test fixture writes; not subject to repository write-token discipline.

    use std::sync::Arc;

    use bytes::Bytes;
    use lore_base::runtime::LORE_CONTEXT;
    use lore_base::runtime::runtime;
    use lore_base::types::Context;
    use lore_revision::branch;
    use lore_revision::immutable;
    use lore_revision::lore::BranchId;
    use lore_revision::lore::RepositoryId;
    use lore_revision::node::Node;
    use lore_revision::node::NodeFlags;
    use lore_revision::node::ROOT_NODE;
    use lore_revision::repository;
    use lore_revision::repository::RepositoryContext;
    use lore_revision::state;
    use lore_storage::hash::hash_string;
    use lore_storage::options::WriteOptions;

    include!("helper.rs");

    #[tokio::test]
    #[allow(clippy::large_futures)]
    async fn node_chain_built_from_facts_survives_serialize_and_fresh_read() {
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

                // Content for the leaf, written directly into the immutable
                // store — no working-tree file ever exists for it, matching
                // the design's "chunk/hash/store the request body directly"
                // step (§3.1 of the doc).
                let content = Bytes::from_static(b"direct-to-store leaf content");
                let context = Context::from(uuid::Uuid::now_v7());
                let address = immutable::write(
                    repository.clone(),
                    context,
                    content.clone(),
                    WriteOptions::default(),
                )
                .await
                .expect("Failed to write leaf content to store");
                let size = content.len() as u64;

                // A brand new, empty in-memory tree -- the equivalent of the
                // staged state a real write path would hold for the
                // lifetime of the serving process (§7 of the doc).
                let tree_state = Arc::new(state::State::new());

                // Walk "a/b/c.txt" from the root, one component at a time,
                // using only caller-supplied facts (name, is-directory, and
                // for the leaf, its already-known content address/size) --
                // never a filesystem stat. This is the doc's §2 "narrower
                // write path into the model that's already there": the
                // low-level primitives (`node_add`) are already `pub` and
                // don't care where the facts came from.
                let dir_a = tree_state
                    .node_add(
                        repository.clone(),
                        ROOT_NODE,
                        Node {
                            name_hash: hash_string("a"),
                            ..Default::default()
                        },
                        "a",
                    )
                    .await
                    .expect("Failed to add directory node 'a'");

                let dir_b = tree_state
                    .node_add(
                        repository.clone(),
                        dir_a,
                        Node {
                            name_hash: hash_string("b"),
                            ..Default::default()
                        },
                        "b",
                    )
                    .await
                    .expect("Failed to add directory node 'b'");

                let leaf = tree_state
                    .node_add(
                        repository.clone(),
                        dir_b,
                        Node {
                            flags: NodeFlags::File.bits(),
                            mode: 0,
                            name_hash: hash_string("c.txt"),
                            size,
                            address,
                            ..Default::default()
                        },
                        "c.txt",
                    )
                    .await
                    .expect("Failed to add leaf node 'c.txt'");

                assert!(
                    tree_state.is_dirty(),
                    "node_add should have dirtied the state"
                );

                // Durability step: serialize the dirty blocks this walk
                // touched (§3.3 of the doc). No batch stage() call anywhere
                // in this test.
                let signature = tree_state
                    .serialize(repository.clone(), &write_token)
                    .await
                    .expect("Failed to serialize tree state");
                assert_ne!(
                    signature,
                    lore_base::types::Hash::default(),
                    "a dirtied, non-empty tree must not serialize to the zero hash"
                );

                // Fresh read: an independent State, deserialized purely from
                // the signature, with none of tree_state's in-memory runtime
                // carried over. Proves the structure is durable and
                // correctly addressed the moment the walk above returned --
                // not merely visible through the same State object.
                let fresh_state = state::State::deserialize(repository.clone(), signature)
                    .await
                    .expect("Failed to deserialize fresh state from signature");

                let node_link = fresh_state
                    .find_node_link(repository.clone(), "a/b/c.txt")
                    .await
                    .expect("Failed to find a/b/c.txt in the fresh read");
                assert_eq!(node_link.node, leaf);

                let leaf_node = fresh_state
                    .node(repository.clone(), node_link.node)
                    .await
                    .expect("Failed to load leaf node from fresh state");
                assert!(leaf_node.is_file());
                assert_eq!(leaf_node.address, address);
                assert_eq!(leaf_node.size, size);

                // Confirm the intermediate directory structure round-tripped
                // too, not just the leaf's own address/size fields.
                let root_children = fresh_state
                    .node_children(repository.clone(), ROOT_NODE)
                    .await
                    .expect("Failed to list root's children in the fresh read");
                assert_eq!(root_children, vec![dir_a]);

                let a_children = fresh_state
                    .node_children(repository.clone(), dir_a)
                    .await
                    .expect("Failed to list 'a's children in the fresh read");
                assert_eq!(a_children, vec![dir_b]);

                let b_children = fresh_state
                    .node_children(repository.clone(), dir_b)
                    .await
                    .expect("Failed to list 'b's children in the fresh read");
                assert_eq!(b_children, vec![leaf]);

                // And that the leaf's content is actually retrievable by the
                // address that made the round trip, not just structurally
                // equal.
                let read_options = immutable::read_options_from_repository(&repository);
                let read_back =
                    immutable::read(repository.clone(), leaf_node.address, None, read_options)
                        .await
                        .expect("Failed to read back leaf content");
                assert_eq!(read_back.as_ref(), content.as_ref());

                let _ = std::fs::remove_dir_all(path.as_path());
            }))
            .await
            .expect("Test task failed");
    }
}
