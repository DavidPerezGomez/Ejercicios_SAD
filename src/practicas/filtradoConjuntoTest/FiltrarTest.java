package practicas.filtradoConjuntoTest;

import java.util.ArrayList;
import java.util.Enumeration;
import java.util.Random;

import weka.attributeSelection.BestFirst;
import weka.attributeSelection.CfsSubsetEval;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Attribute;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.supervised.attribute.AttributeSelection;
import weka.filters.unsupervised.attribute.Remove;

public class FiltrarTest {

	public static void main(String[] args) {
		String path = "./src/files/breast-cancer.arff";

		// cargamos el set completo
		Instances instances = null;
		try {
			DataSource ds = new DataSource(path);
			instances = ds.getDataSet();
		} catch (Exception e) {
			System.out.println("Error al cargar los datos de " + path);
			e.printStackTrace();
			System.exit(1);
		}
		
		// dividimos el set en train y dev
		double trainPercent = 70.0;
		instances.randomize(new Random(1));
		int trainSize = (int) (instances.size()*trainPercent/100);
		int devSize = instances.size() - trainSize;
		Instances train = new Instances(instances, 0, trainSize);
		Instances dev = new Instances(instances, trainSize, devSize);
		System.out.println("Total instances: " + instances.size() + "\nTrain size: " + trainSize + "\nDev size: " + devSize);
		train.setClassIndex(train.numAttributes() - 1);
		dev.setClassIndex(dev.numAttributes() - 1);

		Instances filteredTrain = null, filteredDev = null;
		try {
			Instances[] res = filterFSS(train, dev);
            filteredTrain = res[0];
            filteredDev = res[1];
		} catch (Exception e) {
			System.out.println("Error al filtrar los datos");
			e.printStackTrace();
			System.exit(1);
		}

		// se crea el clasificador
        try {
            NaiveBayes nv = new NaiveBayes();
            Evaluation evaluation = new Evaluation(filteredTrain);
            nv.buildClassifier(filteredTrain);
            evaluation.crossValidateModel(nv, filteredDev, 10, new Random(1));
            System.out.println(evaluation.toSummaryString());
            System.out.println(evaluation.toClassDetailsString());
            System.out.println(evaluation.toMatrixString());
        } catch (Exception e) {
            System.out.println("Error al evaluar el clasificador");
            e.printStackTrace();
            System.exit(1);
        }
    }
	
	private static Instances[] filterFSS(Instances pTrain, Instances pDev) throws Exception {
		// A partir de este punto suponemos que las instancias del conjunto de test (dev)
		// no tienen el valor de la clase asignado, por lo que no podemos utilizarlo.
		// Suponemos también que en este momento train y dev son compatibles y que
		// tienen el índice de la clase correctamente indicado.
		
		// se construye el filtro
		AttributeSelection filter = new AttributeSelection();
		CfsSubsetEval eval = new CfsSubsetEval();
		BestFirst search = new BestFirst();
		filter.setEvaluator(eval);
		filter.setSearch(search);
		filter.setInputFormat(pTrain);
		
		// se filtran los datos de train
		Instances filteredTrain = Filter.useFilter(pTrain, filter);
		
		// se obtienen los atributos de dev y de train filtrado para comparar
		// este metodo devuelve todos los atributos menos la clase
		Enumeration<Attribute> devAttrEnum = pDev.enumerateAttributes();
		Enumeration<Attribute> trainAttrEnum = filteredTrain.enumerateAttributes();

		// se pasan los atributos a una colección porque es más cómodo de manejar
		ArrayList<Attribute> devAttr = new ArrayList<Attribute>();
		ArrayList<Attribute> trainAttr = new ArrayList<Attribute>();
		while(trainAttrEnum.hasMoreElements()) {
			trainAttr.add(trainAttrEnum.nextElement());
		}
		while(devAttrEnum.hasMoreElements()) {
			devAttr.add(devAttrEnum.nextElement());
		}

		// se muestran los atributos por pantalla
		System.out.println("\nAtributos de train (filtrados)");
		for (Attribute attr : trainAttr) {
			System.out.println(attr.name());
		}
		System.out.println("\nAtributos de dev (sin filtrar)");
		for(Attribute attr : devAttr) {
			System.out.println(attr.name());
		}

		// se busca qué atributos de dev no están en train (han sido filtrados)
        // y guardamos sus índices
        int[] indices = new int[devAttr.size() - trainAttr.size()];
        int i = 0;
		System.out.println("\nAtributos que han sido filtrados");
		for(Attribute attr : devAttr) {
		    // el ArrayList hace la comprobación con .equals()
            // los propios atributos se encargan de comprobar si son iguales
			if (!trainAttr.contains(attr)) {
                indices[i] = attr.index();
                System.out.println(indices[i] + ": " + attr.name());
                i++;
			}
		}

		// se construye el filtro Remove
        Remove rmFilter = new Remove();
        rmFilter.setAttributeIndicesArray(indices);
        rmFilter.setInputFormat(pDev);

        // se filtran las instancias de dev para eliminar los
        // atributos que se habían eliminado de train
        Instances filteredDev = Filter.useFilter(pDev, rmFilter);

        // se muestra por pantalla los atributos de dev tras el filtrado
        Enumeration<Attribute> filteredDevAttrEnum = filteredDev.enumerateAttributes();
        System.out.println("\nAtributos de dev (filtrados)");
        while (filteredDevAttrEnum.hasMoreElements()) {
            System.out.println(filteredDevAttrEnum.nextElement().name());
        }

        return new Instances[]{filteredTrain, filteredDev};
	}
	
}
