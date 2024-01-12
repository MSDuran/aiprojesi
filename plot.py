import matplotlib.pyplot as plt

metrics = ['RMSE', 'MAE']
without_bayesnet = [0.927886535251345, 0.7328749729061197]
with_bayesnet = [0.9133509671968227, 0.7235790072330671]

fig, ax = plt.subplots(figsize=(10, 5))

pos = range(len(without_bayesnet))
width = 0.4

ax.barh([p - width/2 for p in pos], without_bayesnet, width, label='Without BayesNet', color='blue')
ax.barh([p + width/2 for p in pos], with_bayesnet, width, label='With BayesNet', color='orange')

ax.set_yticks(pos)
ax.set_yticklabels(metrics)

for i in range(len(metrics)):
    ax.text(without_bayesnet[i], i - width/2, str(without_bayesnet[i]), va='center', ha='right', color='white', fontweight='bold')
    ax.text(with_bayesnet[i], i + width/2, str(with_bayesnet[i]), va='center', ha='right', color='white', fontweight='bold')

plt.legend()
plt.title('Comparison of RMSE and MAE: With and Without BayesNet')

plt.savefig('comparison_rmse_mae.png')
plt.show()
