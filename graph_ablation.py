import matplotlib.pyplot as plt

iterations = [5, 10, 15, 20, 50]
asr_with = [98, 96, 92, 92, 91]
asr_without = [60, 56, 56, 54, 58]

plt.plot(iterations, asr_with, marker='o', label='ASR (with attack soft prompt)')
plt.plot(iterations, asr_without, marker='o', label='ASR (without attack soft prompt)')
plt.xlabel('Iterations')
plt.ylabel('ASR (%)')
plt.title('Iterations vs ASR')
plt.legend()
plt.grid(True)
plt.ylim(0, 100)
plt.tight_layout()
plt.show()