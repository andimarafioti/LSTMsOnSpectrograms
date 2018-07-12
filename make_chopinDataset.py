from datasetGenerator.fmaTFRecordGenerator import FMATFRecordGenerator

__author__ = 'Andres'

from datasetGenerator.exampleProcessor import ExampleProcessor

__author__ = 'Andres'

exampleProcessor = ExampleProcessor(gapLength=8192, sideLength=4096, hopSize=1024, gapMinRMS=1e-3)
tfRecordGenerator = FMATFRecordGenerator(baseName='chopin', pathToDataFolder='utils', exampleProcessor=exampleProcessor)
tfRecordGenerator.generateDataset()
