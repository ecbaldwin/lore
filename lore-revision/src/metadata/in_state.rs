// SPDX-FileCopyrightText: 2026 Epic Games, Inc.
// SPDX-License-Identifier: MIT
//! Cascade fork-local: a node's file metadata, read and replaced against a
//! caller's in-memory [`State`], by node id.
//!
//! `set.rs`'s `set_file` is one whole CLI operation: it resolves a path, loads
//! the staged state itself, reads binary values from files, and serializes and
//! anchors the state when it's done. A server that keeps its own `State` across
//! requests needs only the middle of that: the metadata of a node it has
//! already resolved, and a replacement it built itself. These are that middle,
//! keeping `set_file_task`'s compare-and-swap: a store names the hash it was
//! built from, and leaves a slot that has changed since then alone.
//!
//! A store marks the node staged, as a content change does, so a staged
//! rehash of its ancestors sees the change and a commit records it.

use std::sync::Arc;

use lore_error_set::prelude::*;

use crate::errors::NodeNotFound;
use crate::lore::Hash;
use crate::metadata::Metadata;
use crate::metadata::MetadataError;
use crate::node;
use crate::node::NodeFileMetadata;
use crate::node::NodeFileMetadataBlock;
use crate::node::NodeFlags;
use crate::node::NodeID;
use crate::repository::RepositoryContext;
use crate::state::State;

/// The file metadata `node` carries in `state`, and its hash: zero, with an
/// empty [`Metadata`], when it carries none.
pub async fn file_metadata_load(
    repository: Arc<RepositoryContext>,
    state: &State,
    node: NodeID,
) -> Result<(Hash, Metadata), MetadataError> {
    let hash = file_metadata_hash(repository.clone(), state, node).await?;
    if hash.is_zero() {
        return Ok((hash, Metadata::new()));
    }
    let metadata = Metadata::deserialize(repository, hash).await?;
    Ok((hash, metadata))
}

/// The hash of the file metadata `node` carries in `state`; zero for none.
pub async fn file_metadata_hash(
    repository: Arc<RepositoryContext>,
    state: &State,
    node: NodeID,
) -> Result<Hash, MetadataError> {
    let metadata_node = node::node_to_file_metadata(node);
    let block = state
        .try_block_file_metadata_existing(
            repository,
            NodeFileMetadataBlock::index(metadata_node),
        )
        .await
        .forward_any::<MetadataError>("loading the file metadata block")?;
    Ok(block.map_or_else(Hash::default, |block| {
        block
            .read()
            .node(NodeFileMetadata::index(metadata_node))
            .metadata
    }))
}

/// Replace `node`'s file metadata with `metadata` (an empty one clears it), if
/// the node still carries `expected`, the hash [`file_metadata_load`] returned.
/// Returns the new hash, or `None` when the slot holds something else by now,
/// for the caller to load again and rebuild. A node staged for deletion has
/// no metadata to replace.
pub async fn file_metadata_store(
    repository: Arc<RepositoryContext>,
    state: &State,
    node: NodeID,
    expected: Hash,
    metadata: &Metadata,
) -> Result<Option<Hash>, MetadataError> {
    let target = state
        .node(repository.clone(), node)
        .await
        .forward_any::<MetadataError>("loading the node")?;
    if target.is_staged_delete() {
        return Err(NodeNotFound.into());
    }

    let updated = metadata.serialize(repository.clone()).await?;
    let metadata_node = node::node_to_file_metadata(node);
    let block_index = NodeFileMetadataBlock::index(metadata_node);
    // Clearing a slot in a block that was never stored has nothing to clear.
    let block = if updated.is_zero() {
        state
            .try_block_file_metadata_existing(repository.clone(), block_index)
            .await
    } else {
        state
            .block_file_metadata(repository.clone(), block_index)
            .await
            .map(Some)
    }
    .forward_any::<MetadataError>("loading the file metadata block")?;
    let Some(block) = block else {
        return Ok(expected.is_zero().then_some(updated));
    };

    let dirtied = {
        let mut writer = block.write();
        let slot = writer.node(NodeFileMetadata::index(metadata_node));
        if slot.metadata != expected {
            return Ok(None);
        }
        if slot.metadata == updated {
            return Ok(Some(updated));
        }
        slot.metadata = updated;
        writer.mark_dirty()
    };
    if dirtied {
        state.block_file_metadata_modified(block, block_index);
        state.mark_dirty();
    }

    let flags = if target.is_directory() {
        NodeFlags::Staged
    } else {
        State::staged_edit_flags(&target).0
    };
    state
        .node_mark(repository, node, flags, true)
        .await
        .forward_any::<MetadataError>("marking the node staged")?;
    Ok(Some(updated))
}
