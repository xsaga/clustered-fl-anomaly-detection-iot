df.index = pd.to_datetime(df["time"], unit="s")

ans=df.resample("s")["resultado"].aggregate(np.mean).fillna(0)

#ans[np.isnan(ans)]=0

np.corrcoef(ans, ans)

coco = [1.0]
for i in range(1,200):
    coco.append(np.corrcoef(ans[i:], ans[:-i])[0,1])
plt.plot(coco)
plt.show()
