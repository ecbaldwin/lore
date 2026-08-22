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

use std::sync::Arc;

use crate::errors::InvalidArguments;
use crate::hash;
use crate::lore::Address;
use crate::node::Node;
use crate::node::NodeBlock;
use crate::node::NodeFlags;
use crate::node::NodeID;
use crate::node::ROOT_NODE;
use crate::node::TRUSTED_CONTENT_MARKER;
use crate::repository::RepositoryContext;
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
    let components: Vec<&str> = path.split('/').filter(|c| !c.is_empty()).collect();
    if components.is_empty() {
        return Err(StateError::from(InvalidArguments {
            reason: "cannot stage the repository root as a leaf".into(),
        }));
    }

    let mut parent = ROOT_NODE;
    let last = components.len() - 1;
    for (index, name) in components.iter().enumerate() {
        let is_leaf = index == last;
        parent = stage_one_component(
            repository.clone(),
            state.clone(),
            parent,
            name,
            is_leaf,
            content,
        )
        .await?;
    }

    Ok(parent)
}

async fn stage_one_component(
    repository: Arc<RepositoryContext>,
    state: Arc<State>,
    parent: NodeID,
    name: &str,
    is_leaf: bool,
    content: LeafContent,
) -> Result<NodeID, StateError> {
    let name_hash = hash::hash_string(name);

    match state.find_subnode(repository.clone(), parent, name_hash).await {
        Ok(existing) => {
            let node = state.node(repository.clone(), existing).await?;
            if is_leaf {
                if !node.is_file() {
                    return Err(StateError::from(InvalidArguments {
                        reason: format!(
                            "cannot stage '{name}' as a file: an existing node there is not a file"
                        ),
                    }));
                }
                update_leaf(&state, &repository, existing, content).await?;
                state
                    .node_mark(repository.clone(), existing, NodeFlags::StagedModify, true)
                    .await?;
            } else if !node.is_directory() {
                return Err(StateError::from(InvalidArguments {
                    reason: format!(
                        "cannot stage '{name}' as a directory: an existing node there is not a directory"
                    ),
                }));
            }
            Ok(existing)
        }
        Err(e) if e.is_node_not_found() => {
            let node = if is_leaf {
                Node {
                    flags: NodeFlags::File.bits(),
                    mode: content.mode,
                    name_hash,
                    size: content.size,
                    address: content.address,
                    reserved: TRUSTED_CONTENT_MARKER,
                    ..Default::default()
                }
            } else {
                Node {
                    name_hash,
                    ..Default::default()
                }
            };
            let node_id = state
                .node_add(repository.clone(), parent, node, name)
                .await?;
            let flags = if is_leaf {
                NodeFlags::StagedAdd
            } else {
                NodeFlags::Staged
            };
            state
                .node_mark(repository.clone(), node_id, flags, true)
                .await?;
            Ok(node_id)
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
