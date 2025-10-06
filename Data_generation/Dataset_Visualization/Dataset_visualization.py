import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("C:\Data Gen\Baseline - Data\Phi_0p8_u_0p2_30s_20250118_152820.txt", sep='\t')

plt.figure(figsize=(12, 6))
plt.plot(df.iloc[:, 0], label='Feature 1')
plt.plot(df.iloc[:, 1], label='Feature 2')
plt.plot(df.iloc[:, 2], label='Feature 3')
plt.xlabel('Time Step')
plt.ylabel('Value')
plt.title('Time Series Plot of Data')
plt.legend()
plt.grid(True)
plt.show()


print("Data Shape:")
print(df.shape)

print("\nFirst 5 Rows of Data:")
print(df.head())

print("\nBasic Data Statistics:")
print(df.describe())
