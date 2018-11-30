if [ -d ~/esper ]; then
    # We are not in the container
    ROOT_DIR=~/esper/app
else
    ROOT_DIR=/app
fi
SUBS=${ROOT_DIR}/data/subs/orig
mkdir -p $SUBS
gsutil -m cp gs://esper/movie-metadata/subs/*.srt $SUBS
