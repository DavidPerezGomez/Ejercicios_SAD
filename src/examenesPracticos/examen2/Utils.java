package examenesPracticos.examen2;

import weka.attributeSelection.BestFirst;
import weka.attributeSelection.CfsSubsetEval;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.ConverterUtils;
import weka.filters.Filter;
import weka.filters.supervised.attribute.AttributeSelection;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Random;

/**
 * Esta clase contiene método estáticos para realizar variedad de funciones genéricas e independientes.
 * Para usar los métodos se puede copiar y pegar esta clase en el paquete deseado.
 * A la hora de introducir nuevos métodos, basta con acutalizar la versión de Utils que esté en el ejercicio (examen/práctica) más reciente.
 * Esta es una buena clase para tener disponible en los exámenes de implementación para ahorrar mucho tiempo.
 */
public class Utils {

    /**
     * Evaluación Leave-One-Out
     * Realiza una evaluación Leave-One-Out sobre el clasificador pClassifier con las instancias pIsntaces y la seed pSeed donde.
     *
     * @param pClassifier clasificador a evaluar
     * @param pInstances  instancias con las que evaluar
     * @param pSeed       seed para la randomizanión
     * @return el objeto Evaluation que contiene los resultados de la evaluación, null si hay problemas al evaluar el clasificador.
     */
    public static Evaluation evalLeaveOneOut(Classifier pClassifier, Instances pInstances, long pSeed) {
        Evaluation evaluation = null;
        try {
            evaluation.crossValidateModel(pClassifier, pInstances, pInstances.numInstances(), new Random(pSeed));
        } catch (Exception e) {
            printlnError("Error al evaluar el clasificador");
            e.printStackTrace();
        }
        return evaluation;
    }

    /**
     * Evaluación k-Fold Cross-Validation
     * Realiza una evaluación k-Fold Cross-Validation sobre el clasificador pClassifier con las instancias pIsntaces y la seed pSeed donde k es pFolds.
     *
     * @param pClassifier clasificador a evaluar
     * @param pInstances  instancias con las que evaluar
     * @param pFolds      número de iteraciones a realizar
     * @param pSeed       seed para la randomizanión
     * @return el objeto Evaluation que contiene los resultados de la evaluación, null si hay problemas al evaluar el clasificador.
     */
    public static Evaluation evalKFoldCrossValidation(Classifier pClassifier, Instances pInstances, int pFolds, long pSeed) {
        Evaluation evaluation = null;
        try {
            evaluation.crossValidateModel(pClassifier, pInstances, pFolds, new Random(pSeed));
        } catch (Exception e) {
            printlnError("Error al evaluar el clasificador");
            e.printStackTrace();
        }
        return evaluation;
    }

    /**
     * Evaluación Hold-Out
     * Se dividen las instancias de pInstaces en dos set (train y test) con el tamaño indicado por pTrainPercent.
     * Se entrena el clasificador pClassifier con el set train y se evalua con el set test.
     *
     * @param pClassifier   clasificador a evaluar
     * @param pInstances    instancias con las que evaluar
     * @param pTrainPercent porcentaje de instancias que se usarán para el entrenamiento
     * @return el objeto Evaluation que contiene los resultados de la evaluación, null si hay problemas al evaluar el clasificador.
     */
    public static Evaluation evalHoldOut(Classifier pClassifier, Instances pInstances, double pTrainPercent) {
        int numTrain = (int) (pInstances.numInstances() * pTrainPercent / 100);
        int numTest = pInstances.numInstances() - numTrain;
        pInstances.randomize(new Random(1));
        Instances train = new Instances(pInstances, 0, numTrain);
        Instances test = new Instances(pInstances, numTrain, numTest);
        return evalHoldOut(pClassifier, train, test);
    }

    /**
     * Evaluación Hold-Out
     * Se entrena el clasificador pClassifier con el set pTrain y se evalua con el set pTest.
     *
     * @param pClassifier clasificador a evaluar
     * @param pTrain      instancias que se usarán para el entrenamiento
     * @param pTest       instancias que se usarán para el entrenamiento
     * @return el objeto Evaluation que contiene los resultados de la evaluación, null si hay problemas al evaluar el clasificador.
     */
    public static Evaluation evalHoldOut(Classifier pClassifier, Instances pTrain, Instances pTest) {
        Evaluation evaluation = null;
        try {
            pClassifier.buildClassifier(pTrain);
            evaluation.evaluateModel(pClassifier, pTest);
        } catch (Exception e) {
            printlnError("Error al evaluar el clasificador");
            e.printStackTrace();
        }
        return evaluation;
    }

    /**
     * Evaluación no Honesta
     * Entrena y evalua el clasificador pClassifier con las instancias pInstancias.
     *
     * @param pClassifier clasificador a evaluar
     * @param pInstances  instancias con las que evaluar
     * @return el objeto Evaluation que contiene los resultados de la evaluación, null si hay problemas al evaluar el clasificador.
     */
    public static Evaluation evalWithTrainSet(Classifier pClassifier, Instances pInstances) {
        Evaluation evaluation = null;
        try {
            pClassifier.buildClassifier(pInstances);
            evaluation.evaluateModel(pClassifier, pInstances);
        } catch (Exception e) {
            printlnError("Error al evaluar el clasificador");
            e.printStackTrace();
        }
        return evaluation;
    }

    /**
     * Carga las instancias del archivo en pPath.
     * Se establece pClassIndex como el índice de la clase. -1 indica que la clase es el último atributo.
     *
     * @param pPath       ruta del archivo
     * @param pClassIndex índice del atributo clase
     * @return el objeto de tipo Instances con las instancias del archivo, null si hay problemas al cargar los datos.
     */
    public static Instances loadInstances(String pPath, int pClassIndex) {
        Instances instances = null;
        try {
            ConverterUtils.DataSource ds = new ConverterUtils.DataSource(pPath);
            instances = ds.getDataSet();
            if (pClassIndex >= 0)
                instances.setClassIndex(pClassIndex);
            else
                instances.setClassIndex(instances.numAttributes() - 1);
        } catch (Exception e) {
            printlnError(String.format("Error al cargar las instancias de %s", pPath));
            e.printStackTrace();
        }
        return instances;
    }

    /**
     * Carga las instancias del archivo en pPath.
     * Se supone que la clase es el último atributo.
     *
     * @param pPath ruta del archivo
     * @return el objeto de tipo Instances con las instancias del archivo, null si hay problemas al cargar los datos.
     */
    public static Instances loadInstances(String pPath) {
        return loadInstances(pPath, -1);
    }

    /**
     * Devuelve el índice del valor de la clase mayoritaria en el set pInstances.
     * Solo funciona en sets con clase de tipo nominal.
     *
     * @param pInstances set de instancias
     * @return el índice del valor de la clase mayoritaria.
     */
    public static int getMayorityClassIndex(Instances pInstances) {
        int[] nomCounts = pInstances.attributeStats(pInstances.classIndex()).nominalCounts;
        int maxClassAmount = -1;
        int maxClassIndex = -1;
        for (int i = 0; i < nomCounts.length; i++) {
            if (nomCounts[i] > maxClassAmount) {
                maxClassAmount = nomCounts[i];
                maxClassIndex = i;
            }
        }
        return maxClassIndex;
    }

    /**
     * Devuelve el índice del valor de la clase minoritaria en el set pInstances.
     * Solo funciona en sets con clase de tipo nominal.
     *
     * @param pInstances set de instancias
     * @return el índice del valor de la clase minoritaria.
     */
    public static int getMinorityClassIndex(Instances pInstances) {
        int[] nomCounts = pInstances.attributeStats(pInstances.classIndex()).nominalCounts;
        int minClassAmount = -1;
        int minClassIndex = -1;
        for (int i = 0; i < nomCounts.length; i++) {
            if (minClassAmount < 0 || nomCounts[i] < minClassAmount) {
                minClassAmount = nomCounts[i];
                minClassIndex = i;
            }
        }
        return minClassIndex;
    }

    /**
     * Realiza un filtrado de atributos en las instancias pInstancias.
     * Las instancias pasadas deben tener el índice de la clase ya asignado.
     * Las instancias que se devuelve tienen el atributo clase movido a la última posición por defecto (numInstances()-1).
     *
     * @param pInstances instancias cuyos atributos se quieren filtrar
     * @return instancias con sus atributos filtrados.
     */
    public static Instances attributeSelectionFilter(Instances pInstances) {
        Instances filteredData = null;
        try {
            AttributeSelection filter = new AttributeSelection();
            CfsSubsetEval eval = new CfsSubsetEval();
            BestFirst search = new BestFirst();
            filter.setEvaluator(eval);
            filter.setSearch(search);
            filter.setInputFormat(pInstances);
            filteredData = Filter.useFilter(pInstances, filter);
        } catch (Exception e) {
            printlnError("Error al filtrar los atributos");
            e.printStackTrace();
        }
        return filteredData;
    }

    /**
     * Escribe el texto pTexto en el archivo en la ruta pPath.
     *
     * @param pText texto a escribir
     * @param pPath ruta del archivo
     */
    public static void writeToFile(String pText, String pPath) {
        try {
            BufferedWriter bf = new BufferedWriter(new FileWriter(pPath));
            bf.write(pText);
            bf.close();
        } catch (IOException e) {
            printlnError(String.format("Error al escribir en %s", pPath));
            e.printStackTrace();
        }
    }

    /**
     * Guarda el clasificador pClassifier en la ruta pPath.
     *
     * @param pClassifier clasificador a guardar
     * @param pPath       ruta del archivo a crear
     */
    public static void saveModel(Classifier pClassifier, String pPath) {
        try {
            SerializationHelper.write(pPath, pClassifier);
        } catch (Exception e) {
            printlnError("Error al guardar el clasificador");
            e.printStackTrace();
        }
    }

    /**
     * Carga el clasificador pClassifier desde la ruta pPath.
     *
     * @param pPath ruta del archivo a cargar
     * @return el clasificador cargado.
     */
    public static void loadModel(String pPath) {
        try {
            SerializationHelper.read(pPath);
        } catch (Exception e) {
            printlnError("Error al cargar el clasificador");
            e.printStackTrace();
        }
    }

    /**
     * Escribe por consola el texto pTexto en color rojo.
     * SIEMPRE incluye salto de línea.
     *
     * @param pText texto a escribir
     */
    public static void printlnError(String pText) {
        printError(String.format("%s\n", pText));
    }

    /**
     * Escribe por consola el texto pTexto en color rojo.
     * NO incluye salto de línea.
     *
     * @param pText texto a escribir
     */
    public static void printError(String pText) {
        System.out.print(String.format("\33[31m%s\33[0m", pText));
    }

}
