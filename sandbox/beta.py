
import irispie as ir

x = ir.Series(start_date=ir.qq(2025,1), values=(1,2,3,4))

t = ir.qq(2025,4)

print(t)

# Get values at time t, t-1, t+1
print(x(t))
print(x(t-1))
print(x(t+1))

# Get new time series shifted by -1 or +1
print(x[-1])
print(x[+1])

xm1 = x[-1]
xp1 = x[+1]
print(x | xm1 | xp1)




x = ir.Series(start_date=ir.qq(2020,1), values=(1,2,3,4))
y = ir.Series(start_date=ir.qq(2025,1), values=(1,2,3,4))

print(x(ir.start))
print(y(ir.start))

print(ir.start.resolve(x))
print(ir.start.resolve(y))

