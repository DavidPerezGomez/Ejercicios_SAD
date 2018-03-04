package practicas.practica3;
import weka.attributeSelection.BestFirst;
import weka.attributeSelection.CfsSubsetEval;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.lazy.IBk;
import weka.core.*;
import weka.filters.Filter;
import weka.filters.supervised.attribute.AttributeSelection;

import java.util.Random;

public class KNNParameterEval {

    // variables de instancia
    NormalizableDistance[] posDValues;
    int[] posWValues;

    /**
     * Constructora de la clase
     */
    public KNNParameterEval() {
        // valores que pueden tomar los parámetros d y w
        posDValues = new NormalizableDistance[] {new EuclideanDistance(), new ManhattanDistance()};
        posWValues = new int[] {IBk.WEIGHT_NONE, IBk.WEIGHT_INVERSE, IBk.WEIGHT_SIMILARITY};
    }

    /**
     * Método para determinar los valores k, d y w (o cualquier combinación de ellos) más apropiados para clasificar el conjunto de datos dato mediante IBk.
     * @param pPath dirección del archivo .arff que contiene los datos
     * @param pClassIndex índice del atributo clase
     * @param pFilter true si se quiere pasar los datos por un filtro. false en caso contrario
     * @param pEvalK true si se quiere evaluar los valores del parámetro k. false en caso contrario
     * @param pEvalD true si se quiere evaluar los valores del parámetro d. false en caso contrario
     * @param pEvalW true si se quiere evaluar los valores del parámetro w. false en caso contrario
     * @param pVerbose  true si se quiere que se muestre por pantalla el resultado de cada combinación de los tres parámetros. false en caso contrario
     */
    public void evaluateParameters(String pPath, int pClassIndex, boolean pFilter, boolean pEvalK, boolean pEvalD, boolean pEvalW, boolean pVerbose) {
        // se cargan los datos del archivo
        Instances initialData = InstancesLoader.load(pPath);
        if (initialData != null) {

            // se indica el índice de la clase
            initialData.setClassIndex(pClassIndex);

            // se filtran o no los atributos según el input
            Instances filteredData;
            if (pFilter) {
                filteredData = filterData(initialData);
            } else {
                filteredData = initialData;
            }

            // variables para almacenar los posibles valores de cada atributo
            int kMaxValue;
            NormalizableDistance[] dValues;
            int[] wValues;

            double fMeasure;

            // se inicializa el valor máximo que va a poder tomar k
            if(pEvalK)
//                kMaxValue = filteredData.numInstances() - 1;
                kMaxValue = Math.min(filteredData.numInstances() - 1, 100);
            else
                kMaxValue = 1;

            // se inicializan los valores que va a poder tomar d
            if(pEvalD)
                dValues = posDValues;
            else
                dValues = new NormalizableDistance[] {posDValues[0]};

            // se inicializan los valores que va a poder tomar w
            if(pEvalW)
                wValues = posWValues;
            else
                wValues = new int[] {posWValues[0]};

            // variable para guardar el mejor resultado en cualquier momento dado
            IBkResultSet topResult = null;

            // se itera anidadamente sobre todos los posible valores de k, d y w
            for (int k = 1; k <= kMaxValue; k++) {
                for (NormalizableDistance d : dValues) {
                    for (int w : wValues) {

                        // se crea el clasificador IBk con los parámetros k, d y w correspondientes
                        IBk classifier = createIBkClassifier(k, d, w);

                        // se evalua el clasificador
                        Evaluation evaluation = evaluateClassifier(classifier, filteredData);

                        // se obtiene el f-measure de la evaluación
                        fMeasure = findFMeasure(evaluation, filteredData);

                        IBkResultSet result = new IBkResultSet(k, d, w, fMeasure);

                        // si el resultado mejor que el mejor hasta el momento,
                        // se guarda como nuevo mejor resultado
                        if (topResult == null || fMeasure > topResult.getFMeasure())
                            topResult = result;

                        if (pVerbose) {
                            System.out.println(result.toString() + "\n");
                        }
                    }
                }
            }
            System.out.println("################################");
            System.out.println("MEJOR RESULTADO:");
            System.out.println(topResult.toString());
            // se hace de nuevo la evaluación
            IBk classifier = createIBkClassifier(topResult.getK(), topResult.getD(), topResult.getW());
            Evaluation evaluation = evaluateClassifier(classifier, filteredData);
            printEvaluation(evaluation);
        }
    }

    /**
     * Dada la evaluación de un clasificador calcula la media ponderada de los f-measure de cada valor de la clase.
     * Hace lo mismo que Evaluation.weightedFMeasure() con la excepción de que, si el f-measure de alguno de los
     * valores de la clase es NaN, se tomará como 0 en lugar de devolver un f-measure medio de NaN.
     *
     * @param pEvaluation
     * @param pData datos con los que ha sido obtenida la evaluación
     * @return el f-measure del clasificador
     */
    private double findFMeasure(Evaluation pEvaluation, Instances pData) {
        double fMeasure = Double.NaN;
        try {
            Attribute classAtr = pData.classAttribute();

            // array con el f-measure de cada valor de la clase
            // en caso de que alguno de los f-measure fueran NaN se toma como 0
            // ya que significa que tanto precision como recall son 0
            double[] fMeasures = new double[classAtr.numValues()];
            for (int i = 0; i < classAtr.numValues(); i++) {
                fMeasures[i] = pEvaluation.fMeasure(i);
                if (Double.isNaN(fMeasures[i])) {
                    fMeasures[i] = 0;
                }
            }

            // array con el peso de cada valor de la clase
            // como peso se utiliza el numero de instancias del conjunto de datos
            // que presentan esa clase
            int[] weights = new int[classAtr.numValues()];
            for (int j = 0; j < pData.numInstances(); j++) {
                int classValue = (int) pData.instance(j).classValue();
                weights[classValue]++;
            }

            // se calcula f-measure como la media ponderada de los f-measure de cada valor de la clase
            fMeasure = 0;
            for (int i = 0; i < classAtr.numValues(); i++) {
                fMeasure += weights[i]*fMeasures[i];
            }
            fMeasure = fMeasure/pData.numInstances();
        } catch (Exception e) {
            System.out.println("\033[31mError al calcular el f-measure.\033[0m");
        }
        return fMeasure;
    }

    /**
     * Se evalua el clasificador pClassifier con los datos pData utilizando
     * el método 10-fold cross-validation.
     * @param pClassifier
     * @param pData
     * @return el resultado de la evaluación
     */
    private Evaluation evaluateClassifier(Classifier pClassifier, Instances pData) {
        Evaluation evaluator = null;
        try {
            evaluator = new Evaluation(pData);
            int folds = 10;
            evaluator.crossValidateModel(pClassifier, pData, folds, new Random(1));
        } catch (Exception e) {
            System.out.println("\033[31mError al evaluar el clasificador.\033[0m");
            evaluator = null;
        }
        return evaluator;
    }

    /**
     * Se crea un clasificador tipo IBk con los parámetros pK, pD y pW dados.
     * @param pK (k) numero de vecinos que se consulta en el algoritmo KNN (<= 1)
     * @param pD (d) método de obtener la distancia entre dos atributos
     * @param pW (w) método de calcular el peso del "voto" de un vecino dada su distancia (1, 2 ó 4)
     * @return el clasificador IBk sin entrenar
     */
    private IBk createIBkClassifier(int pK, NormalizableDistance pD, int pW) {
        IBk classifier = null;
        try {
            classifier = new IBk();
            classifier.setKNN(pK);
            classifier.getNearestNeighbourSearchAlgorithm().setDistanceFunction(pD);
            classifier.setDistanceWeighting(new SelectedTag(pW, IBk.TAGS_WEIGHTING));
        } catch (Exception e) {
            System.out.println("Error al construir el clasificador.");
            e.printStackTrace();
            System.exit(1);
        }
        return classifier;
    }

    /**
     * Filtra los atributos de las instancias de pData.
     * @param pData
     * @return las instancias de pData con sus atributos filtrados
     */
    private Instances filterData(Instances pData) {
        try {
            AttributeSelection filter = new AttributeSelection();
            CfsSubsetEval eval = new CfsSubsetEval();
            BestFirst search = new BestFirst();
            filter.setEvaluator(eval);
            filter.setSearch(search);
            filter.setInputFormat(pData);
            return Filter.useFilter(pData, filter);
        } catch (Exception e) {
            e.printStackTrace();
            return null;
        }
    }

    /**
     * Muestra por consola los resultados de la evaluación dada.
     * @param pEvaluation
     */
    private void printEvaluation(Evaluation pEvaluation) {
        try {
            System.out.println(pEvaluation.toSummaryString());
            System.out.println(pEvaluation.toClassDetailsString());
            System.out.println(pEvaluation.toMatrixString());
        } catch (Exception e) {
            System.out.println("\033[31mError al mostrar la información de la evaluación\033[0m");
            System.out.println(e.getMessage());
        }
    }

    /**
     * Método para determinar el valor k más apropiado
     * @param pPath
     * @param pClassIndex
     * @param pFilter
     * @param pVerbose
     */
    public void evaluateKParameter(String pPath, int pClassIndex, boolean pFilter, boolean pVerbose) {
        evaluateParameters(pPath, pClassIndex, pFilter, true, false, false, pVerbose);
    }

    /**
     * Método para determinar los valores k, d y w más apropiados
     * @param pPath
     * @param pClassIndex
     * @param pFilter
     * @param pVerbose
     */
    public void evaluateAllParameters(String pPath, int pClassIndex, boolean pFilter, boolean pVerbose) {
        evaluateParameters(pPath, pClassIndex, pFilter, true, true, true, pVerbose);
    }

}
