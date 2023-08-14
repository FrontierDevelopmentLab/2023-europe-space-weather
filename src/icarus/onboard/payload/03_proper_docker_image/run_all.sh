source /opt/intel/openvino/bin/setupvars.sh -pyver 3.7

echo ""
echo "--- COMPRESSOR MODEL ---"
#python3.7 payload/openvino_use_compressor.py

# two images:
# cor1_20090822_130000_s4c1a.fts
# cor2_20090611_125300_n4c2a.fts
echo "COR 1 file (0.5MB, 512x512 resolution)"
python3.7 payload/openvino_use_compressor.py --input data/cor1_20090822_130000_s4c1a.fts --bottleneck_name "latent_cor1_512_"
echo "COR 2 file (8MB, 2048x2048 resolution)"
python3.7 payload/openvino_use_compressor.py --input data/cor2_20090611_125300_n4c2a.fts --model weights/onboard_compressor_y_2048.onnx --resolution 2048  --bottleneck_name "latent_cor2_2048_"
