// SPDX-FileCopyrightText: 2026 Epic Games, Inc.
// SPDX-License-Identifier: MIT
//
// Cascade fork-local: checkpoint 2 of
// `docs/proposed/direct-to-store-writes.md` in the cascade `server` repo --
// "implement §4's eager bubble-up (directory address recomputed on every
// write, from the leaf's parent to the root)."
//
// Key finding this test establishes: no new fork code is needed for the
// hashing itself. `commit::rehash_directory` (`lore-revision/src/
// commit.rs:2696`) -- already `pub`, already the exact function `commit()`
// itself calls at `commit.rs:1521` to compute directory Merkle addresses --
// only recomputes a directory node whose `NodeFlags::Staged` bit is set (or
// the root, unconditionally), and recurses into staged directory children
// only. `state::node_mark` (also already `pub`) walks a leaf's ancestor
// chain to the root marking each one `Staged` as a side effect of marking
// the leaf itself. Composing those two existing primitives -- call
// `node_mark` on the leaf, then `rehash_directory` on the root -- reproduces
// exactly the "recompute only the touched ancestors" behavior §4 asks for,
// driven per-write instead of only from `commit()`. §4's "this is cheap, not
// merely acceptable" argument (live sibling reads, no rehashing of
// unrelated content) is a description of what `rehash_directory` already
// does, not a new algorithm to build.
#[cfg(test)]
mod tests {
    #![allow(clippy::disallowed_methods)] // Test fixture writes; not subject to repository write-token discipline.

    use std::io::Write;
    use std::sync::Arc;

    use bytes::Bytes;
    use lore_base::runtime::LORE_CONTEXT;
    use lore_base::runtime::runtime;
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
    use lore_revision::node::Node;
    use lore_revision::node::NodeFlags;
    use lore_revision::node::ROOT_NODE;
    use lore_revision::repository;
    use lore_revision::repository::RepositoryContext;
    use lore_revision::stage;
    use lore_revision::stage::StageOptions;
    use lore_revision::state;
    use lore_storage::hash::hash_string;
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

    #[tokio::test]
    #[allow(clippy::large_futures)]
    async fn eager_bubble_up_matches_commit_time_directory_addresses() {
        let execution = setup_test_execution();

        #[allow(clippy::disallowed_methods)]
        runtime()
            .spawn(LORE_CONTEXT.scope(execution.clone(), async move {
                let leaf_content = Bytes::from_static(b"direct-to-store leaf content, bubble-up test");

                // --- Side A: eager, per-write bubble-up. No commit() call
                // anywhere in this half -- the tree is built the same way
                // checkpoint 1's PoC did (node_add from explicit facts, no
                // lore_io), then the leaf's ancestor chain is rehashed
                // immediately, the way §4 wants every write to behave.
                let (immutable_a, mutable_a, _) =
                    test_store_create().await.expect("Failed to create stores (A)");
                let tempdir_a = generate_tempdir();
                std::fs::create_dir_all(tempdir_a.path()).expect("Create directory failed (A)");
                let (repository_a, write_token_a) =
                    new_repository(immutable_a, mutable_a, tempdir_a.path()).await;

                let context = Context::from(uuid::Uuid::now_v7());
                let address = immutable::write(
                    repository_a.clone(),
                    context,
                    leaf_content.clone(),
                    WriteOptions::default(),
                )
                .await
                .expect("Failed to write leaf content to store (A)");
                let size = leaf_content.len() as u64;

                let tree_state = Arc::new(state::State::new());

                let dir_a = tree_state
                    .node_add(
                        repository_a.clone(),
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
                        repository_a.clone(),
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
                        repository_a.clone(),
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

                // Bubbles NodeFlags::Staged up the leaf's ancestor chain
                // (dir_b, dir_a, root) as a side effect of marking the leaf
                // itself -- see state::node_mark's ancestor-climbing loop.
                tree_state
                    .node_mark(
                        repository_a.clone(),
                        leaf,
                        NodeFlags::StagedAdd,
                        true, /* mark dirty */
                    )
                    .await
                    .expect("Failed to mark leaf staged");

                // The eager step §4 asks for: recompute directory addresses
                // now, from the leaf's parent to the root, instead of
                // waiting for commit(). rehash_staged_directory only
                // recomputes nodes still carrying the Staged bit node_mark
                // just set, so this only touches dir_a/dir_b/root -- nothing
                // unrelated. Unlike commit()'s own rehash_directory, this
                // fork-local variant tolerates (expects) the leaf still
                // being marked staged, since nothing has been committed yet.
                commit::rehash_staged_directory(repository_a.clone(), tree_state.clone(), ROOT_NODE)
                    .await
                    .expect("Failed to eagerly rehash directory ancestors");

                let signature = tree_state
                    .serialize(repository_a.clone(), &write_token_a)
                    .await
                    .expect("Failed to serialize eagerly-rehashed tree");

                // Fresh, independent read -- proves the bubble-up produced
                // durable, correctly-addressed directory nodes, not values
                // only visible through tree_state's own runtime cache.
                let fresh_state = state::State::deserialize(repository_a.clone(), signature)
                    .await
                    .expect("Failed to deserialize fresh state (A)");
                let eager_dir_a = fresh_state
                    .node(repository_a.clone(), dir_a)
                    .await
                    .expect("Failed to load dir 'a' from fresh state");
                let eager_dir_b = fresh_state
                    .node(repository_a.clone(), dir_b)
                    .await
                    .expect("Failed to load dir 'a/b' from fresh state");
                assert!(
                    !eager_dir_a.address.hash.is_zero(),
                    "eager bubble-up must produce a real Merkle hash for 'a', not a placeholder"
                );
                assert!(
                    !eager_dir_b.address.hash.is_zero(),
                    "eager bubble-up must produce a real Merkle hash for 'a/b', not a placeholder"
                );

                let _ = std::fs::remove_dir_all(tempdir_a.path());

                // --- Side B: today's actual commit() pipeline, as ground
                // truth. Same structure, same leaf content, written to real
                // files and staged/committed the normal way -- no fork
                // mechanism involved on this side at all.
                let (immutable_b, mutable_b, _) =
                    test_store_create().await.expect("Failed to create stores (B)");
                let tempdir_b = generate_tempdir();
                std::fs::create_dir_all(tempdir_b.path()).expect("Create directory failed (B)");
                let (repository_b, write_token_b) =
                    new_repository(immutable_b, mutable_b, tempdir_b.path()).await;

                let dir_path = tempdir_b.path().join("a").join("b");
                std::fs::create_dir_all(&dir_path).expect("Failed to create a/b on disk");
                let file_path = dir_path.join("c.txt");
                write_test_file(file_path.as_path(), leaf_content.as_ref());

                // Content addressing pairs a Hash (content) with a Context
                // (file identity/lineage), and stage_node_from_metadata
                // mints a fresh random Context for every normally-staged
                // file. Left as `trusted_content: None`, side A and side B's
                // leaves would get two different, independently-generated
                // Contexts -- and since a directory's Merkle hash is over
                // its children's whole Address (hash *and* context, per
                // `NodeHashData`), that alone would make the two sides'
                // directory hashes diverge for a reason that has nothing to
                // do with whether eager bubble-up is correct. Reuse the
                // exact (address, size) side A already wrote so both sides'
                // leaf nodes carry an identical Address, making the
                // directory-hash comparison below actually meaningful.
                immutable::write(
                    repository_b.clone(),
                    context,
                    leaf_content.clone(),
                    WriteOptions::default(),
                )
                .await
                .expect("Failed to write matching leaf content to store (B)");

                file::stage::stage(
                    repository_b.clone(),
                    &write_token_b,
                    LoreArray::from_vec(vec![LoreString::from(tempdir_b.path())]),
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
                .expect("Failed to stage the real on-disk tree");

                // Regression guard: StageOptions::trusted_content vouches
                // for one file's content address. stage_node_from_metadata
                // used to apply it unconditionally in its "create new node"
                // branch, so a fresh ancestor directory created in the same
                // stage() call as the trust-staged leaf (as "a" and "a/b"
                // are here) would get stamped with the leaf's own
                // (address, size) too -- corrupting its later Merkle hash.
                // Caught by this test before it could reach the directory-
                // hash comparison below; fixed by gating the override on
                // `node.is_file()` (stage.rs).
                {
                    let staged_signature = lore_revision::instance::load_staged_revision(&repository_b)
                        .await
                        .expect("Failed to load staged revision")
                        .expect("Expected a staged revision after stage()");
                    let staged_state = state::State::deserialize(repository_b.clone(), staged_signature)
                        .await
                        .expect("Failed to deserialize staged state (B)");
                    let staged_dir_b_link = staged_state
                        .find_node_link(repository_b.clone(), "a/b")
                        .await
                        .expect("Failed to find staged 'a/b'");
                    let staged_dir_b = staged_state
                        .node(repository_b.clone(), staged_dir_b_link.node)
                        .await
                        .expect("Failed to load staged 'a/b'");
                    assert!(
                        staged_dir_b.address.hash.is_zero() && staged_dir_b.address.context.is_zero(),
                        "trusted_content must not leak onto a freshly-created ancestor \
                         directory; got address {:?}",
                        staged_dir_b.address
                    );
                }

                let commit_options = CommitOptions {
                    message: String::new(),
                    link_messages: std::collections::HashMap::new(),
                    link: None,
                    layer_messages: std::collections::HashMap::new(),
                    layer: None,
                };
                let commit_signature =
                    Box::pin(commit::commit(repository_b.clone(), &write_token_b, commit_options))
                        .await
                        .expect("Failed to commit the real on-disk tree");

                let committed_state = state::State::deserialize(repository_b.clone(), commit_signature)
                    .await
                    .expect("Failed to deserialize committed state (B)");
                let committed_dir_a_link = committed_state
                    .find_node_link(repository_b.clone(), "a")
                    .await
                    .expect("Failed to find committed 'a'");
                let committed_dir_a = committed_state
                    .node(repository_b.clone(), committed_dir_a_link.node)
                    .await
                    .expect("Failed to load committed 'a'");
                let committed_dir_b_link = committed_state
                    .find_node_link(repository_b.clone(), "a/b")
                    .await
                    .expect("Failed to find committed 'a/b'");
                let committed_dir_b = committed_state
                    .node(repository_b.clone(), committed_dir_b_link.node)
                    .await
                    .expect("Failed to load committed 'a/b'");
                let committed_leaf_link = committed_state
                    .find_node_link(repository_b.clone(), "a/b/c.txt")
                    .await
                    .expect("Failed to find committed 'a/b/c.txt'");
                let committed_leaf = committed_state
                    .node(repository_b.clone(), committed_leaf_link.node)
                    .await
                    .expect("Failed to load committed 'a/b/c.txt'");

                // The whole point: recomputing directory addresses eagerly,
                // per-write, produces bit-identical results to computing
                // them lazily at commit() -- same content, same structure,
                // same Merkle hash, regardless of when the hashing ran.
                assert_eq!(committed_leaf.address, address);
                assert_eq!(committed_leaf.size, size);
                assert_eq!(
                    eager_dir_b.address, committed_dir_b.address,
                    "eager and commit-time Merkle hashes for 'a/b' must match"
                );
                assert_eq!(
                    eager_dir_a.address, committed_dir_a.address,
                    "eager and commit-time Merkle hashes for 'a' must match"
                );

                let _ = std::fs::remove_dir_all(tempdir_b.path());
            }))
            .await
            .expect("Test task failed");
    }
}
