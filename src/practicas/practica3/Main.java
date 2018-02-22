package practicas.practica3;

public class Main {

    public static void main(String[] args) {
        KNNParameterEval parameterEval = new KNNParameterEval();
//        parameterEval.evaluateKParameter("./src/files/balance-scale.arff", 4, false, true);
//        parameterEval.evaluateKParameter("./src/files/wine.arff", 0, true, true);
//        parameterEval.evaluateAllParameters("./src/files/balance-scale.arff", 4, false, true);
        parameterEval.evaluateAllParameters("./src/files/wine.arff", 0, true, true);
    }
}
