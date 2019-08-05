# FAQs

In this document, we list common questions that users may ask about simulating their applications.
### My simulation takes a very long time to run. What should I do?
This is likely due to the fact that your input data is large. Simulating an application on large data usually takes a while. You can either leave the process running and wait for the result, or you can try simulating your application on smaller data.

If you are building a graph-based application, you can try to simulate the application with a smaller graph. For example, NetworkX provides functions to generate random social graph at arbitrary size:
```python
import networkx as nx

# Create a caveman graph with 1 clique and 500 nodes in the clique
g = nx.caveman_graph(1, 500)
```

If you are building an neural network application, you can try to simulate the application with a smaller feature size or hidden size.

Then, you can linearly scale the time required for a larger dataset based on the numbers obtained for the smaller ones. The power number should be roughly the same across different data sizes.

### My Python script terminates with a `Killed` message. What should I do? 
This is likely due to the fact that your kernel issued an OOM killer. You can either run your simulation with a more powerful machine, or you should try to clean up your system memory before doing the simulation.
