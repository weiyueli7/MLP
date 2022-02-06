mkdir -p data/train data/test

curl -o data/train/images.npz -L http://codh.rois.ac.jp/kmnist/dataset/kmnist/kmnist-train-imgs.npz
curl -o data/train/labels.npz -L http://codh.rois.ac.jp/kmnist/dataset/kmnist/kmnist-train-labels.npz
curl -o data/test/images.npz -L http://codh.rois.ac.jp/kmnist/dataset/kmnist/kmnist-test-imgs.npz
curl -o data/test/labels.npz -L http://codh.rois.ac.jp/kmnist/dataset/kmnist/kmnist-test-labels.npz