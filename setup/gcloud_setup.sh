# Source> https://medium.com/@jamsawamsa/running-a-google-cloud-gpu-for-fast-ai-for-free-5f89c707bae6
# Source2> https://blog.kovalevskyi.com/deep-learning-images-for-google-cloud-engine-the-definitive-guide-bc74f5fb02bc
# Make sure to have Owner rights> https://console.cloud.google.com/iam-admin/iam?_ga=2.192088234.-1861512181.1540122253&_gac=1.255463290.1540122651.CjwKCAjwx7DeBRBJEiwA9MeX_K8D6d1KNBFSuhnWXyieUhKZT-AyjBHLKJoa-CEJusZOMdGSy15qxhoCKk0QAvD_BwE
# Image types available> https://cloud.google.com/deep-learning-vm/docs/images
# Machine types> https://cloud.google.com/compute/docs/machine-types
# Zones> https://cloud.google.com/compute/docs/regions-zones/
export IMAGE_FAMILY="<image_name>"
export ZONE="<zone>"
export INSTANCE_NAME=$1
export INSTANCE_TYPE="<instnce_type>"
gcloud compute instances create $INSTANCE_NAME \
        --zone=$ZONE \
        --image-family=$IMAGE_FAMILY \
        --image-project=deeplearning-platform-release \
        --maintenance-policy=TERMINATE \
        --accelerator='type=nvidia-tesla-k80,count=1' \
        --machine-type=$INSTANCE_TYPE \
        --boot-disk-size=120GB \
        --metadata='install-nvidia-driver=True'
# Access via ssh:
# gssh $INSTANCE_NAME -- -L 8080:localhost:8080
