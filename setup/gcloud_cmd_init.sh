#!/bin/bash
ginstance_start() { 
  gcloud compute instances start $@
}

ginstance_stop() { 
  gcloud compute instances stop $@
}

gssh() { 
  gcloud compute ssh $@
}

glab() {
    gssh $@ -- -L 8080:localhost:8080
}

# e.g.: gcloud compute instances start jku-project
# e.g.: gcloud compute ssh jku-project -- -L 8080:localhost:8080
# e.g.: gcloud compute instances stop jku-project

# add function to the environment by callsing> . gcloud_cmd_init.sh
# source> https://blog.kovalevskyi.com/deep-learning-images-for-google-cloud-engine-the-definitive-guide-bc74f5fb02bc