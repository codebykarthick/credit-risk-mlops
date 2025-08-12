cd src/

# Uncomment the lines to do the required process.
# This does the split and imputes the values
# python3 data_preprocessing.py all
# This only splits the data into train, test and val, to explore train before imputing.
# python3 data_preprocessing.py split
# This imputes the datasets to make it ready training
python3 data_preprocessing.py impute

# The below adds some features that I think will at least make it easier to explain the performance
python3 features.py