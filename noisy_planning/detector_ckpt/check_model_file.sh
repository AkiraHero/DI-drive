#!/bin/bash
echo "dd49ed43989e818f180fa5476f543aa28f14e0ec12ce6a7f07def37fcce469e7 pointpillar_checkpoint_epoch_120.pth" |sha256sum --check
echo "2a1a3511a89793618524ad94bf10e9f1c57fc8c49f5fa49adf08fa0e94974d8d  pvrcnn_checkpoint_epoch_120.pth" |sha256sum --check
echo "2338efdd7ef860b79633fdd5bc3ed7d68f725cc7e16d3c88e4bec60803b597dc centerpoint_checkpoint_epoch_120.pth"  |sha256sum --check