package practicas.practica3;

import java.util.Scanner;

public class Main {

    public static void main(String[] args) {
        if(args.length == 0)
            interactiveRun();
    }

    private static void interactiveRun() {
        Scanner scanner = new Scanner(System.in);
        String tmp;
        System.out.print("Introduce la ruta del arhivo .arff: ");
        String path = scanner.nextLine();
        System.out.print("Introduce el índice del atributo clase (empezando por 0): ");
        int classIndex = scanner.nextInt();
        scanner.nextLine(); //para consumir el new-line character que nextInt() no lee
        System.out.print("¿Quieres realizar un filtrado de atributos? [S/n]: ");
        tmp = scanner.nextLine();
        boolean filter = tmp.toLowerCase().charAt(0) == 's';
        System.out.print("¿Quieres que se evalúe el parámetro k? [S/n]: ");
        tmp = scanner.nextLine();
        boolean k = tmp.toLowerCase().charAt(0) == 's';
        System.out.print("¿Quieres que se evalúe el parámetro d? [S/n]: ");
        tmp = scanner.nextLine();
        boolean d = tmp.toLowerCase().charAt(0) == 's';
        System.out.print("¿Quieres que se evalúe el parámetro w? [S/n]: ");
        tmp = scanner.nextLine();
        boolean w = tmp.toLowerCase().charAt(0) == 's';
        System.out.print("¿Quieres que se muestre el resultado de cada comprobación por consola? [S/n]: ");
        tmp = scanner.nextLine();
        boolean verbose = tmp.toLowerCase().charAt(0) == 's';

        KNNParameterEval parameterEval = new KNNParameterEval();
        parameterEval.evaluateParameters(path, classIndex, filter, k, d, w, verbose);
    }

}
