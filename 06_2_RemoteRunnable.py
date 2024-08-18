from langserve import RemoteRunnable

llama2=RemoteRunnable('http://localhost:9000/llama2/')
response=llama2.stream({'input':'Hi there.'})

chunks=[]
resp=''
for chunk in response:
    chunks.append(chunk)
    print(chunk,end=' ', flush=True)
print()
for i in range(len(chunks)):
    resp+=chunks[i]
print(resp)