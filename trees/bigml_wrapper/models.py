#
# BigML public decision tree models and datasets
#
from feature import *

models = {
    'iris':      "565b275f55b0c80b6b00459c",    # iris dataset
    'election':  "50a7a9d4035d0706dc0004e8",    # white-box elections
    'cars':      "50f2d9cb3b56354d2c0002d2",    # car-crash
    'military':  "54467d0b99fca41102000e83",    # military-gear
    'steak':     "537be4c3d9949779990041e9",    # steak survey          => GOOD!
    'diabetes':  "515d7f78035d07412f0001a8",    # diabetes
    'telephony': "53a071d4c8db6379930014f6",    # telephony
    'credit':    "5655b4b73cd257732f0050ae",    # german credit
    'gss':       "5124321c035d071fdc000fda",    # GSS happiness survey  => GOOD!
    'email':     "52fa8aa60c0b5e6d4a0030c1",    # email importance      => good
    'spam':      "56a09a9d200d5a337100d4b9",    # spam email            => good
    'tax':       "50c008bb3b56351981000375",    # tax patterns          => GOOD!
    'medical':   "51a37e0e925ded36f3000629",    # medical provider
    'movies':    "565723c21d550571a7011895",    # X-rated movies
    'digits':    "52c1c3960c0b5e6fc900016d",    # mnist digits
    'med_price': "519e2666925ded779c000db7",    # medical coverage      => GOOD! (with eps=1e-3)
    'bitcoin':   "51ccb90a035d07603900039e"     # bitcoin price         => GOOD! (with eps=1e-4)
}

black_box_models = {
    'credit': "56ab34f01d550529ca00e9ae",
    'steak': "537be4c3d9949779990041e9"
}

#
# Download datasets using:
# curl "https://bigml.io/public/dataset/{DATA_ID}/download?$BIGML_AUTH" > ../data/bigml_datasets/{DATA_ID}.csv
#

datasets = {
    'tax':          "50c008453b5635197f000295",
    'gss':          "512431e73b56352c74000b08",
    'spam':         "507a389e035d07092100008e",
    'email':        "52fa89390c0b5e6d480020d9",
    'bitcoin':      "51ccb8f9035d076038000290",
    'med_price':    "519e25c9925ded7798001542",
    'credit':       "4f89c38f1552686459000033"
}