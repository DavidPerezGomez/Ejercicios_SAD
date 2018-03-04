package practicas.practica3;

import weka.classifiers.lazy.IBk;
import weka.core.NormalizableDistance;

public class IBkResultSet {

        // variables de instancia
        int k;
        NormalizableDistance d;
        int w;
        double fMeasure;

        /**
         * Constructora de la clase
         * @param pK
         * @param pD
         * @param pW
         * @param pFMeasure
         */
        public IBkResultSet(int pK, NormalizableDistance pD, int pW, double pFMeasure) {
            k = pK;
            d = pD;
            w = pW;
            fMeasure = pFMeasure;
        }

        /**
         * Recoge los datos de la instancia en una String que se puede mostrar por pantalla.
         * @return
         */
        public String toString() {
            StringBuilder str = new StringBuilder();
            str.append("k: " + k + "\n");
            str.append("d: " + d.getClass().toString() + "\n");
            str.append("w: ");
            switch (w) {
                case IBk.WEIGHT_NONE:
                    str.append("No distance weighting");
                    break;
                case IBk.WEIGHT_INVERSE:
                    str.append("Weight by 1/distance");
                    break;
                case IBk.WEIGHT_SIMILARITY:
                    str.append("Weight by 1 - distance");
                    break;
                default:
                    break;
            }
            str.append("\n");
            str.append("f-measure: " + fMeasure);
            return str.toString();
        }

        public int getK() {
            return k;
        }

        public NormalizableDistance getD() {
            return d;
        }

        public int getW() {
            return w;
        }

        public double getFMeasure() {
            return fMeasure;
        }
    }
