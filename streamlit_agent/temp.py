import datetime
import numpy as np

start = datetime.date(2024, 3, 31)
end = datetime.date(2024, 1, 1)

days = np.busday_count(start, end)
days*8*120
