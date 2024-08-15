import torch
import igraph as ig
from dataloader import DataLoader
import os
import statics


class GraphExplorer:
    """
    A class that constructs an igraph.Graph object and provides visualization and
    metric calculation functionality.
    """
    dataloader: DataLoader
    graph: ig.Graph
    train_only: bool

    def __init__(self, dataloader: DataLoader, train_only: bool = None):
        """
        The constructor
        :param dataloader: The dataloader containing the graph triples
        :param train_only: Use only training triples to create the Graph
        """
        self.train_only = train_only if train_only is not None else False
        self.dataloader = dataloader
        if self.dataloader.num_nodes_total <= 0:
            raise Exception("The Dataloader object cannot be empty.")
        self.save_path = os.path.join(os.path.join(os.getcwd(), "data"), "Visualizations")
        statics.check_and_make_dir(self.save_path)
        self.create_graph()

    def create_graph(self):
        """
        A function that creates an iGraph Graph object from the triples in the dataset
        :return:None
        """
        if type(self.dataloader.train) != torch.Tensor or type(self.dataloader.validation) != torch.Tensor \
                or type(self.dataloader.test) != torch.Tensor:
            raise Exception("The triple data must be in torch.Tensor format.")
        if self.train_only:
            heads, relations, tails = self.dataloader.train[:, 0].numpy(), self.dataloader.train[:, 1].numpy(), \
                                      self.dataloader.train[:, 2].numpy()
            edges = [[h, t] for h, t in zip(heads, tails)]
            self.graph = ig.Graph(n=statics.count_unique_nodes(self.dataloader.train), edges=edges,
                                  edge_attrs={'edge_type': list(relations)},
                                  directed=True)
        else:
            triples = torch.cat((self.dataloader.train, self.dataloader.validation, self.dataloader.test), dim=0)
            heads, relations, tails = triples[:, 0].numpy(), triples[:, 1].numpy(), triples[:, 2].numpy()
            edges = [[h, t] for h, t in zip(heads, tails)]
            self.graph = ig.Graph(n=self.dataloader.num_nodes_total, edges=edges,
                                  edge_attrs={'edge_type': list(relations)}, directed=True)

    def calculate_metrics(self):
        pass

    def visualize_graph(self, layout: str = None):
        """
        A function that creates a visualization of the graph and saves it as a png file
        :param layout: The selected layout. For the supported layouts check the iGraph documentation
        https://igraph.readthedocs.io/en/stable/tutorial.html#layouts-and-plotting
        :return:None
        """
        layout = self.graph.layout("drl") if layout is None else self.graph.layout(layout)
        ig.plot(self.graph, layout=layout).save(fname=os.path.join(self.save_path, self.dataloader.dataset + ".png"))
