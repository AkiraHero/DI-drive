#!/bin/bash
echo "8acd9b6a3aea7b1f772219a4269bf627752c4994c2532e164a589fa988a81bb0 pointpillar_checkpoint_epoch_146.pth" |sha256sum --check
echo "f63a80bb3c949c851de45c844a9f787946fa0b86de64e75af5ef191d47a311d9  pvrcnn_checkpoint_epoch_171.pth" |sha256sum --check
echo "022ce4ffb9bc671746a2bf0ffe2ed380ee4941a01eccd4faa614cfad7a243b3f centerpoint_checkpoint_epoch_156.pth"  |sha256sum --check
echo "529ac4af8b5cb5e4411f06a27e26958422e0d26843136df737744fd6bebe8775 baseline_pp.pt"  |sha256sum --check
echo "ccf13277d89cb142892c37562ee058c524f0ce0aa8802074436046a8e495c535 baseline_pvrcnn.pt"  |sha256sum --check
echo "baf7a07226afc6f31d97517bb4e31a02eeeb97901380bde16935f6ff873a6d87 baseline_cp.pt"  |sha256sum --check
