package practicas.practica3;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

/**
 * clase abstracta para leer los datos de un archivo arff
 */
public abstract class InstancesLoader {

    /**
     * Lee los datos del archivo situado en pPath.
     * @param pPath
     * @return los datos del archivo
     */
    public static Instances load(String pPath) {
        try {
            DataSource source = new DataSource(pPath);
            Instances data = source.getDataSet();
            if (data.classIndex() == -1) {
                data.setClassIndex(data.numAttributes()-1);
            }
            return data;
        } catch (Exception e) {
            System.out.println("Error al leer los datos de " + pPath);
            e.printStackTrace();
            return null;
        }
    }

}
