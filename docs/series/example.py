
import sys
sys.path.append("../../..")

from modiphy.dataman.series import Series
from modiphy.dataman.dates import yy, hh, qq, mm, dd, ii

x = Series()
x[ qq(2021,1)>>qq(2024,4) ] = [ i**2 for i in range(16) ]


y = Series(num_columns=2)
y[ qq(2020,1)>>qq(2025,4), 0 ] = 100
y[ qq(2025,1)>>qq(2027,4), 1 ] = -100

print(x)
print(y)
print(y(-1)) # Matlab y{-1}
print(x | y) # Matlab [x, y]
print(y(-1) | 3) # Matlab [y{-1}, 3]

data1 = 0.01 * y[..., 1] # Matlab 0.01 * y{:, 1}
print(data1)

data = 0.01 * y[..., ...] # Matlab 0.01 * y{:, :}
print(data)

print(y.start_date)
print(y.end_date)
print(y.range)

range = y.range
print(range.to_plotly_dates())

