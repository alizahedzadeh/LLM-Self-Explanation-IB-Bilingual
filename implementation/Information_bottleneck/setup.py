import os
import json
import numpy as np
import pandas as pd
import time
from typing import List, Dict, Tuple, Optional, Union, Any
import requests
import re
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# For evaluation metrics
from sklearn.metrics import accuracy_score

# For API rate limiting
import time
from functools import wraps