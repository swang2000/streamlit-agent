import ssl 
import certifi 
from urllib.request import urlopen 

request = "https://nd-123-456-789.p2pify.com/901c7d18b72538fd3324248e1234" 
urlopen(request, context=ssl.create_default_context(cafile=certifi.where()))


import ssl
context = ssl._create_unverified_context()
urlopen(req,context=context)

import ssl
ssl._create_default_https_context = ssl._create_unverified_context
urlopen('https://google.com').read()