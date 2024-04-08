echo "----------------------- Downloading pretrained model -----------------------"

model=$1

if [[ $model == "celebahq" ]]
then
  cp_link="https://www.robots.ox.ac.uk/~vgg/research/derender3d/data/celebahq.pth"
  cp_download_path="results/models/celebahq_nr/checkpoint005.pth"
elif [[ $model == "co3d" ]]
then
  cp_link="https://www.robots.ox.ac.uk/~vgg/research/derender3d/data/co3d.pth"
  cp_download_path="results/models/co3d/checkpoint010.pth"
else
  echo Unknown model: $model
  echo Possible options: \"celebahq\", \"co3d\"
  exit
fi

scriptdir=$(dirname $0)
basedir=$(dirname $scriptdir)
outdir=$(dirname $cp_download_path)

cd $basedir || exit
echo Operating in \"$(pwd)\".
echo Creating directories.
mkdir -p $outdir
echo Downloading checkpoint from \"$cp_link\" to \"$cp_download_path\".
wget -O $cp_download_path $cp_link