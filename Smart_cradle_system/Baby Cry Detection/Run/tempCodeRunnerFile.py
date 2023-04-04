
    feats = feature('rec.wav')
    d=np.zeros((64,12))
    for i in range(len(feats)):
        d[i:,]=feats[i]
    x=np.expand_dims(d,axis=0)    