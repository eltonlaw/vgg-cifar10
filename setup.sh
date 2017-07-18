# Downlaod dataset
wget "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
# Unzip downloaded dataset
tar zxvf cifar-10-python.tar.gz

# Create logs directory
mkdir logs

# Install dependencies with pip
virtualenv venv
source venv/bin/activate
pip3 install -r requirements.txt

# Clean Up
rm cifar-10-python.tar.gz 
