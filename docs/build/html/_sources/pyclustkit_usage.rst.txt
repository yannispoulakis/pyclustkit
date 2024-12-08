pyclustkit usage
==================

Pyclustkit has been designed to be as easy to use out of the box as possible. In this section we are to showcase a couple
of examples to present CVI calculation and meta-feature extraction.

First let's produce some synthetic data

.. code-block:: python

    from sklearn.datasets import make_moons

    x,y = make_moons(n_samples=100, n_features=2)

Calculating Meta-Features
-------------------------

Now to calculate the meta-features we need to import the class from the **metalearning** sub-modules.

.. code-block:: python

    from pyclustkit.metalearning import MFExtractor

    mfe = MFExtractor(x)
    mfe.calculate_mf()

It is as simple as that but you can also limit the meta-features to be calculated by category, name or paper they have
been proposed

.. code-block:: python

    # by name
    mfe.calculate_mf(name="log2_no_instances")

    #You can retrieve all the meta-features along with their meta-data with
    for mf in mfe.meta_features:
        print(mfe.meta_features[mf])

    # by category
    mfe.calculate_mf(category="descriptive")

    # by paper
    mfe.calculate_mf(included_in="Ferrari")


Calculating CVI
-------------------------
Calculating cluster validity indices is as easy

.. code-block:: python
    from pyclustkit.eval import CVIToolbox

    cvit = CVIToolbox(x,y)
    cvit.calculate_icvi()

    # We can either pass a list of a subset of CVI
    cvit.calculate_icvi(cvi=["dunn", "silhouette"])

    # You can see the complete list of CVI with :
    print(list(cvit.methods_list.keys()))


    # We  can also exclude CVI in case the list is smaller than those to include.
    cvit.calculate_icvi(exclude=["dunn", "silhouette"])





.. toctree::
   :maxdepth: 1


   pyclustkit_usage

