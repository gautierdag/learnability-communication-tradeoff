from setuptools import setup


setup(
    name='cldfbench_mollica202Xgforms',
    py_modules=['cldfbench_mollica202Xgforms'],
    include_package_data=True,
    zip_safe=False,
    entry_points={
        'cldfbench.dataset': [
            'mollica202Xgforms=cldfbench_mollica202Xgforms:Dataset',
        ]
    },
    install_requires=[
        'cldfbench',
    ],
    extras_require={
        'test': [
            'pytest-cldf',
        ],
    },
)
