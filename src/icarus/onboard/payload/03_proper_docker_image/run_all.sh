source /opt/intel/openvino/bin/setupvars.sh -pyver 3.7

cd payload
echo ""
echo "--- FULL MODEL ---"
python3.7 /payload/openvino_use_full_model.py

echo ""
echo "--- COMPRESSOR MODEL ---"
python3.7 /payload/openvino_use_compressor.py
