sudo yum install python36 python36-virtualenv python36-pip
alias python=python3
curl -O https://bootstrap.pypa.io/get-pip.py
python3 get-pip.py --user
export PATH=~/.local/bin:$PATH
source ~/.bash_profile
pip install awsebcli --upgrade --user
pip install numpy --user
pip install pandas --user
pip install scipy --user
aws configure
aws s3 ls
aws s3 cp s3://msds697-project/small_train.csv small_train.csv
pip install fastprogress --user
