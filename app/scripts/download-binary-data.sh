if [ -d ~/esper ]; then
    # We are not in the container
    ROOT_DIR=~/esper/app
else
    ROOT_DIR=/app
fi
DATA=${ROOT_DIR}/data
mkdir -p $DATA

cd $DATA

# Download face embeddings
gsutil -m cp gs://esper/movie-metadata/movie-embeddings.tar.gz .
tar xvzf movie-embeddings.tar.gz

# Download face landmarks
gsutil -m cp gs://esper/movie-metadata/movie-face-landmarks.tar.gz .
tar xvzf movie-face-landmarks.tar.gz

# Download poses
gsutil -m cp gs://esper/movie-metadata/movie-poses.tar.gz .
tar xvzf movie-poses.tar.gz

