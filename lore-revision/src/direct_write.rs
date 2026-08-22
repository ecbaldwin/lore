// SPDX-FileCopyrightText: 2026 Epic Games, Inc.
// SPDX-License-Identifier: MIT
//! Cascade fork-local: metadata-free single-path staging.
//!
//! Checkpoint 3 of `docs/proposed/direct-to-store-writes.md` (cascade
//! `server` repo, §2/§5): `stage_filesystem_path`/`stage_node_from_metadata`
//! (`stage.rs`) walk a path root-to-leaf, but every component -- including
//! every ancestor directory -- is discovered by calling
//! `lore_io::IoDriver::global().metadata(...)` against a real filesystem
//! path. There is no existing entry point that builds the same tree
//! structure from caller-supplied facts instead.
//!
//! [`stage_leaf_from_facts`] is that entry point, narrowed to exactly what a
//! whole-file write needs: every ancestor is a directory (created if
//! missing, reused if present), and the leaf is a file whose content
//! address/size/mode the caller already knows (it already wrote the bytes
//! into the immutable store -- see `immutable::write`/`write_from_file`).
//! Reuses the same tree-mutation primitives `stage_node_from_metadata`
//! itself calls (`State::find_subnode`, `State::node_add`), so slot
//! allocation, sibling linking and nametable bookkeeping are exactly the
//! tested logic already relied on elsewhere -- only the source of each
//! component's facts changes.
//!
//! **Deliberately narrower than `stage_node_from_metadata`.** No case-
//! mismatch handling, no link crossing, no layers, no filters, and no
//! type-mismatch delete-and-recreate dance: a path component that already
//! exists as the wrong kind (a file where a directory is expected, or vice
//! versa) is a hard error here, not an automatic reconciliation. None of
//! those apply to a write path that always knows its own exact target path
//! and never scans a real filesystem tree; see the doc's §2 finding that
//! reusing `stage_node_from_metadata` wholesale would mean reimplementing
//! that reconciliation logic independently anyway, which this avoids simply
//! by not needing it.
//!
//! Checkpoint 4 adds three more metadata-free entry points for the same
//! reason -- [`stage_directory_from_facts`] (`create_dir`),
//! [`stage_delete_from_facts`] (`delete`) and [`stage_move_from_facts`]
//! (`rename`) -- plus one correctness fix all four now share: a path
//! component this walk finds already staged-delete must be *undeleted*
//! (its staged-delete flags cleared) rather than treated as a live
//! occupant, since an eager `delete()` immediately preceding a re-create at
//! the same path is now a mainline sequence, not a rare race. A staged-
//! delete node whose *type* doesn't match what's being staged is instead
//! left alone (its deletion stands) and a fresh node is added alongside it,
//! matching `stage_node_from_metadata`'s own type-mismatch delete-and-
//! recreate handling (`stage.rs:1614-1629`).

use std::sync::Arc;

use lore_error_set::ForwardStrict;

use crate::errors::InvalidArguments;
use crate::errors::NodeNotFound;
use crate::hash;
use crate::lore::Address;
use crate::node::Node;
use crate::node::NodeBlock;
use crate::node::NodeFlags;
use crate::node::NodeID;
use crate::node::ROOT_NODE;
use crate::node::TRUSTED_CONTENT_MARKER;
use crate::repository::RepositoryContext;
use crate::stage::StageError;
use crate::stage::StageStats;
use crate::state::State;
use crate::state::StateError;

/// A leaf file's content facts, already known to the caller because it
/// already wrote the bytes into the immutable store -- see
/// `immutable::write`/`write_from_file`.
#[derive(Debug, Clone, Copy)]
pub struct LeafContent {
    pub address: Address,
    pub size: u64,
    pub mode: u16,
}

/// What the final path component of a metadata-free walk is being staged
/// as -- an intermediate component (reached via [`stage_leaf_from_facts`]'s
/// or [`stage_directory_from_facts`]'s ancestor walk) is always a directory
/// regardless of this; this only describes the walk's own target.
#[derive(Debug, Clone, Copy)]
enum ComponentKind {
    Directory,
    File(LeafContent),
}

/// Walk `path` from the repository root, creating any missing ancestor
/// directory and creating-or-updating the leaf file, without ever touching
/// a real filesystem path. Returns the leaf's `NodeID`.
///
/// Every ancestor component must already be (or be freshly created as) a
/// directory, and the final component must already be (or be freshly
/// created as) a file; a path component of the wrong kind is rejected with
/// [`StateError::internal`] rather than reconciled automatically (see the
/// module doc).
///
/// Marks the leaf `NodeFlags::StagedAdd` (new leaf) or `NodeFlags::
/// StagedModify` (existing leaf) via `State::node_mark`, which as a side
/// effect marks every ancestor `NodeFlags::Staged` too -- the walk this
/// crate's real staging code already relies on to know which directories
/// have something staged underneath them (see `commit::rehash_staged_directory`,
/// direct-to-store-writes.md §4/§7).
pub async fn stage_leaf_from_facts(
    repository: Arc<RepositoryContext>,
    state: Arc<State>,
    path: &str,
    content: LeafContent,
) -> Result<NodeID, StateError> {
    stage_path_from_facts(repository, state, path, ComponentKind::File(content)).await
}

/// Walk `path` from the repository root, creating any missing ancestor
/// directory and creating-or-reusing the leaf itself as an empty directory
/// -- the metadata-free equivalent of `create_dir`/MKCOL. Returns the
/// leaf directory's `NodeID`.
///
/// Same reconciliation rules as [`stage_leaf_from_facts`]: every component,
/// including the leaf, must already be (or be freshly created as) a
/// directory.
pub async fn stage_directory_from_facts(
    repository: Arc<RepositoryContext>,
    state: Arc<State>,
    path: &str,
) -> Result<NodeID, StateError> {
    stage_path_from_facts(repository, state, path, ComponentKind::Directory).await
}

async fn stage_path_from_facts(
    repository: Arc<RepositoryContext>,
    state: Arc<State>,
    path: &str,
    leaf: ComponentKind,
) -> Result<NodeID, StateError> {
    let components: Vec<&str> = path.split('/').filter(|c| !c.is_empty()).collect();
    if components.is_empty() {
        return Err(StateError::from(InvalidArguments {
            reason: "cannot stage the repository root as a leaf".into(),
        }));
    }

    let mut parent = ROOT_NODE;
    let last = components.len() - 1;
    for (index, name) in components.iter().enumerate() {
        let kind = if index == last {
            leaf
        } else {
            ComponentKind::Directory
        };
        parent = stage_one_component(repository.clone(), state.clone(), parent, name, kind).await?;
    }

    Ok(parent)
}

/// Removes every staged-delete flag from `node_id`, in place -- the
/// metadata-free equivalent of `stage_node_from_metadata`'s "Undelete if
/// deleted" branch (`stage.rs:1766-1778`), minus the `StagedModify`
/// `node_mark` call that branch also makes: callers here always follow up
/// with their own `node_mark` (a leaf's, whose bubble-up already covers
/// every ancestor this function might be called for).
async fn clear_staged_delete(
    state: &Arc<State>,
    repository: &Arc<RepositoryContext>,
    node_id: NodeID,
) -> Result<(), StateError> {
    let block_index = NodeBlock::index(node_id);
    let block = state.block(repository.clone(), block_index).await?;
    let dirtied = {
        let mut block_writer = block.write();
        block_writer.node(Node::index(node_id)).clear_staged_flags();
        block_writer.mark_dirty()
    };
    if dirtied {
        state.block_modified(block, block_index);
        state.mark_dirty();
    }
    Ok(())
}

async fn create_new_component(
    repository: Arc<RepositoryContext>,
    state: Arc<State>,
    parent: NodeID,
    name: &str,
    name_hash: u64,
    kind: ComponentKind,
) -> Result<NodeID, StateError> {
    let node = match kind {
        ComponentKind::Directory => Node {
            name_hash,
            ..Default::default()
        },
        ComponentKind::File(content) => Node {
            flags: NodeFlags::File.bits(),
            mode: content.mode,
            name_hash,
            size: content.size,
            address: content.address,
            reserved: TRUSTED_CONTENT_MARKER,
            ..Default::default()
        },
    };
    let node_id = state
        .node_add(repository.clone(), parent, node, name)
        .await?;
    let flags = match kind {
        ComponentKind::Directory => NodeFlags::Staged,
        ComponentKind::File(_) => NodeFlags::StagedAdd,
    };
    state
        .node_mark(repository.clone(), node_id, flags, true)
        .await?;
    Ok(node_id)
}

async fn stage_one_component(
    repository: Arc<RepositoryContext>,
    state: Arc<State>,
    parent: NodeID,
    name: &str,
    kind: ComponentKind,
) -> Result<NodeID, StateError> {
    let name_hash = hash::hash_string(name);

    match state
        .find_subnode(repository.clone(), parent, name_hash)
        .await
    {
        Ok(existing) => {
            let node = state.node(repository.clone(), existing).await?;
            let wants_directory = matches!(kind, ComponentKind::Directory);
            let type_matches = if wants_directory {
                node.is_directory()
            } else {
                node.is_file()
            };

            if node.is_staged_delete() {
                if !type_matches {
                    // The deleted node stays deleted; a fresh node of the
                    // right kind is added alongside it, same as
                    // `stage_node_from_metadata`'s type-mismatch handling.
                    return create_new_component(repository, state, parent, name, name_hash, kind)
                        .await;
                }
                clear_staged_delete(&state, &repository, existing).await?;
            } else if !type_matches {
                let expected = if wants_directory {
                    "a directory"
                } else {
                    "a file"
                };
                return Err(StateError::from(InvalidArguments {
                    reason: format!(
                        "cannot stage '{name}' as {expected}: an existing node there is not {expected}"
                    ),
                }));
            }

            if let ComponentKind::File(content) = kind {
                update_leaf(&state, &repository, existing, content).await?;
                state
                    .node_mark(repository.clone(), existing, NodeFlags::StagedModify, true)
                    .await?;
            }
            Ok(existing)
        }
        Err(e) if e.is_node_not_found() => {
            create_new_component(repository, state, parent, name, name_hash, kind).await
        }
        Err(e) => Err(e),
    }
}

/// Update an existing leaf's content facts in place, the way
/// `State::node_modify` does for `mode`/`size`/`address` -- plus setting
/// `reserved` to the trusted-content marker, which `node_modify` doesn't
/// touch (it predates this fork's trust mechanism and is used by other
/// callers that don't want it set). See `stage.rs`'s `StageOptions::
/// trusted_content` for the marker's meaning at commit time.
async fn update_leaf(
    state: &Arc<State>,
    repository: &Arc<RepositoryContext>,
    node_id: NodeID,
    content: LeafContent,
) -> Result<(), StateError> {
    let block_index = NodeBlock::index(node_id);
    let block = state.block(repository.clone(), block_index).await?;
    let dirtied = {
        let mut block_writer = block.write();
        let node = block_writer.node(Node::index(node_id));
        let file_id = node.address.context;
        node.mode = content.mode;
        node.size = content.size;
        node.address = content.address;
        if node.address.context.is_zero() {
            node.address.context = file_id;
        }
        node.reserved = TRUSTED_CONTENT_MARKER;
        block_writer.mark_dirty()
    };
    if dirtied {
        state.block_modified(block, block_index);
        state.mark_dirty();
    }
    Ok(())
}

/// Resolve `path` to a node and stage it (and, for a directory, its whole
/// subtree) as deleted, without touching a real filesystem path -- the
/// metadata-free equivalent of `delete()`/WebDAV `DELETE`.
///
/// Returns `Ok(None)`, not an error, when `path` doesn't resolve to a live
/// node (never staged, already staged-delete, or the path plain doesn't
/// exist) -- matching the WebDAV `DELETE` contract this backs, which is
/// idempotent: deleting an absent path is not an error (see
/// `crates/cascade-webdav/src/handler/delete.rs`, which calls unconditionally
/// whenever no conditional header is present).
pub async fn stage_delete_from_facts(
    repository: Arc<RepositoryContext>,
    state: Arc<State>,
    path: &str,
) -> Result<Option<NodeID>, StageError> {
    let link = match state.find_node_link(repository.clone(), path).await {
        Ok(link) if link.is_valid() => link,
        Ok(_) => return Ok(None),
        Err(e) if e.is_node_not_found() => return Ok(None),
        Err(e) => return Err(e).forward::<StageError>("resolving delete target"),
    };
    let node = state
        .node(repository.clone(), link.node)
        .await
        .forward::<StageError>("reading delete target node")?;
    if node.is_staged_delete() {
        return Ok(None);
    }

    crate::stage::stage_delete(
        repository,
        state,
        link.node,
        NodeFlags::empty(),
        Arc::new(StageStats::default()),
        None,
    )
    .await?;
    Ok(Some(link.node))
}

/// Split `path`'s final component off from its parent path (`""` for a
/// repository-root parent).
fn split_leaf(path: &str) -> (&str, &str) {
    let trimmed = path.trim_matches('/');
    trimmed.rsplit_once('/').unwrap_or(("", trimmed))
}

/// Reparent and/or rename the node at `from_path` to `to_path`, creating any
/// missing ancestor directory under `to_path` and staging any existing node
/// already at `to_path` as deleted (overwrite semantics -- WebDAV `MOVE`'s
/// caller, `crates/cascade-webdav/src/handler/copy_move.rs`, has already
/// checked the `Overwrite` header before ever calling `rename()`), without
/// touching a real filesystem path. Returns the moved node's `NodeID` (a
/// move keeps the node's identity -- see `State::move_node`).
///
/// `from_path` must resolve to a live (non-staged-delete) node; this is the
/// metadata-free equivalent of `rename()`/WebDAV `MOVE` for a source that
/// already has one, which is the only case this is called for (see
/// `direct_write.rs`'s `rename` in the `cascade-fs-lore` crate).
pub async fn stage_move_from_facts(
    repository: Arc<RepositoryContext>,
    state: Arc<State>,
    from_path: &str,
    to_path: &str,
) -> Result<NodeID, StageError> {
    let from_link = state
        .find_node_link(repository.clone(), from_path)
        .await
        .forward::<StageError>("resolving move source")?;
    if !from_link.is_valid() {
        return Err(NodeNotFound.into());
    }
    let from_node = state
        .node(repository.clone(), from_link.node)
        .await
        .forward::<StageError>("reading move source node")?;
    if from_node.is_staged_delete() {
        return Err(NodeNotFound.into());
    }

    if let Ok(to_link) = state.find_node_link(repository.clone(), to_path).await
        && to_link.is_valid()
    {
        let to_node = state
            .node(repository.clone(), to_link.node)
            .await
            .forward::<StageError>("reading move destination node")?;
        if !to_node.is_staged_delete() {
            crate::stage::stage_delete(
                repository.clone(),
                state.clone(),
                to_link.node,
                NodeFlags::empty(),
                Arc::new(StageStats::default()),
                None,
            )
            .await?;
        }
    }

    let (to_parent_path, to_name) = split_leaf(to_path);
    let to_parent = if to_parent_path.is_empty() {
        ROOT_NODE
    } else {
        stage_directory_from_facts(repository.clone(), state.clone(), to_parent_path)
            .await
            .forward::<StageError>("ensuring move destination parent")?
    };

    state
        .move_node(repository.clone(), from_link.node, to_parent, to_name)
        .await
        .forward::<StageError>("moving node")?;

    Ok(from_link.node)
}
