# 1. Create + activate (Python 3.10, matching your local 3.10.16)
conda create -n snac-pack-refactor python=3.10 -y
conda activate snac-pack-refactor

# 2. Install the strictly-pinned TF stack FIRST (order matters — pins must win)
pip install "tensorflow==2.15.1" "keras==2.15.0" "tensorboard==2.15.2" \
            "protobuf==4.25.8" "ml-dtypes==0.3.2" \
            "tensorflow-estimator==2.15.0" "tensorflow-io-gcs-filesystem==0.37.1" \
            "tensorflow-model-optimization==0.7.5"

# 3. Then the rest
pip install -r requirements.txt