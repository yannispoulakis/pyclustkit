from pyclustkit.eval.core._common_processes import *

operations_per_cvi = {'dunn_index':['pairwise_distances, min_inter_cluster_distance','max_intra_cluster_distance'],
                      'gdi53':['cluster_centers','distances_from_cluster_center']}


common_operations = {'x': {'value': None,
                           'requires': []},
                     'y': {'value': None,
                           'requires': []},

                     'data_center': {'value': None,
                                     'changes_on_clusters': False,
                                     'requires': ['x'],
                                     'method': data_center},
                     'distances_from_data_center': {'value': None,
                                                    'changes_on_clusters': False,
                                                    'requires': ['x', 'data_center'],
                                                    'method': distances_from_data_center},

                     'cluster_centers': {'value': None,
                                         'changes_on_clusters': True,
                                         'requires': ['x', 'y'],
                                         'method': cluster_centers},
                     'pairwise_cluster_centers_distances': {'value': None,
                                                            'changes_on_clusters': False,
                                                            'requires': ['cluster_centers'],
                                                            'method': pairwise_cluster_centers_distances},
                     'distances_from_cluster_centers': {'value': None,
                                                        'changes_on_clusters': False,
                                                        'requires': ['x', 'y', 'cluster_centers'],
                                                        'method': distances_from_cluster_centers},
                     'sum_distances_from_cluster_centers':{'value': None,
                                                        'changes_on_clusters': False,
                                                        'requires': ['distances_from_cluster_centers'],
                                                        'method': sum_distances_from_cluster_centers},
                     'cluster_centers_from_data_center_distances': {'value': None,
                                                     'changes_on_clusters': True,
                                                     'requires': ['cluster_centers', 'data_center'],
                                                     'method': cluster_centers_from_data_center},

                     'distances_from_other_cluster_centers': {'value': None,
                                                              'changes_on_clusters': False,
                                                              'requires': ['x', 'y', 'cluster_centers'],
                                                              'method': distances_from_other_cluster_centers},
                     'pairwise_sum_distances_from_cluster_centers': {'value': None,
                                                                     'changes_on_clusters': False,
                                                                     'requires': ['sum_distances_from_cluster_centers'],
                                                                     'method':
                                                                         pairwise_sum_distances_from_cluster_centers},
                     'cluster_centers_from_data_center': {'value': None,
                                                          'changes_on_clusters': True,
                                                          'requires': ['cluster_centers', 'data_center'],
                                                          'method': cluster_centers_from_data_center},

                     'pairwise_distances': {'value': None,
                                            'changes_on_clusters': False,
                                            'requires': ['x'],
                                            'method': distances},
                     'sum_pairwise_distances': {'value': None,
                                                'changes_on_clusters': False,
                                                'requires': ['pairwise_distances'],
                                                'method': sum_distances},

                     'inter_cluster_distances': {'value': None,
                                                    'changes_on_clusters': True,
                                                    'requires': ['pairwise_distances','y'],
                                                    'method':   inter_cluster_distances},
                     'intra_cluster_distances': {'value': None,
                                                    'changes_on_clusters': True,
                                                    'requires': ['pairwise_distances','y'],
                                                    'method': intra_cluster_distances},
                     'max_inter_cluster_distances': {'value': None,
                                                    'changes_on_clusters': True,
                                                    'requires': ['inter_cluster_distances'],
                                                    'method': max_cdistances},
                     'max_intra_cluster_distances': {'value': None,
                                                   'changes_on_clusters': True,
                                                   'requires': ['intra_cluster_distances'],
                                                   'method': max_cdistances},
                     'min_inter_cluster_distances': {'value': None,
                                                    'changes_on_clusters': True,
                                                    'requires': ['inter_cluster_distances'],
                                                    'method': min_cdistances},
                     'min_intra_cluster_distances': {'value': None,
                                                     'changes_on_clusters': True,
                                                     'requires': ['intra_cluster_distances'],
                                                     'method': min_cdistances},
                     'sum_inter_cluster_distances': {'value': None,
                                                     'changes_on_clusters': True,
                                                     'requires': ['inter_cluster_distances'],
                                                     'method': sum_inter_cluster_distances},
                     'sum_intra_cluster_distances': {'value': None,
                                                     'changes_on_clusters': True,
                                                     'requires': ['intra_cluster_distances'],
                                                     'method': sum_intra_cluster_distances},




                     'pairwise_hausdorff': {'value': None,
                                                                     'changes_on_clusters': True,
                                                                     'requires': ['x', 'y'],
                                                                     'method':
                                                                         pairwise_hausdorff},


                     # Scatter Matrices & Dispersion
                     'between_group_scatter_matrix': {'value': None,
                                                      'changes_on_clusters': True,
                                                      'requires': ['x', 'y'],
                                                      'method': between_group_scatter_matrix},
                     'between_group_dispersion': {'value': None,
                                                  'changes_on_clusters': True,
                                                  'requires': ['between_group_scatter_matrix'],
                                                  'method': trace},
                     'within_group_scatter_matrices': {'value': None,
                                                       'changes_on_clusters': True,
                                                       'requires': ['x', 'y','cluster_centers'],
                                                       'method': within_group_scatter_matrices},
                     'within_group_scatter_matrix': {'value': None,
                                                     'changes_on_clusters': True,
                                                     'requires': ['x', 'within_group_scatter_matrices'],
                                                     'method': total_within_group_scatter_matrix},
                     'within_group_dispersion': {'value': None,
                                                 'changes_on_clusters': True,
                                                 'requires': ['within_group_scatter_matrix'],
                                                 'method': trace},
                     'total_scatter_matrix': {'value': None,
                                                      'changes_on_clusters': False,
                                                      'requires': ['x'],
                                                      'method': total_scatter_matrix},

                     # S
                     's_values': {'value': None,
                                           'changes_on_clusters': True,
                                           'requires': ['x', 'y', 'pairwise_distances'],
                                            'method': return_s}
                     }



# Define the adgraph
process_adg = {i: j['requires'] for i, j in common_operations.items() if 'requires' in j.keys()}
