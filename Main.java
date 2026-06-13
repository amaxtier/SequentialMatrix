import java.util.Arrays;
import java.util.Locale;
import java.util.Random;

// ============================================================================
// CLASE ABSTRACTA: define la estructura común para cualquier tipo de matriz
// ============================================================================
abstract class Matrix {
    protected int[][] matrixData;
    protected int columnDimension;
    protected int rowDimension;
    protected Random rObject;
    protected static final int RANDOM_SCALE = 10;

    public Matrix() {}

    // Constructor que genera una matriz aleatoria con valores en [-10, 10]
    public Matrix(int rDimension, int cDimension) {
        rObject = new Random(System.currentTimeMillis());
        this.rowDimension = rDimension;
        this.columnDimension = cDimension;
        matrixData = new int[this.rowDimension][this.columnDimension];

        for (int i = 0; i < rowDimension; i++) {
            for (int j = 0; j < columnDimension; j++) {
                matrixData[i][j] = rObject.nextInt(RANDOM_SCALE * 2) - RANDOM_SCALE;
            }
        }
    }

    // Constructor que recibe datos ya definidos (útil para pruebas)
    public Matrix(int rDimension, int cDimension, int[][] mData) {
        this.rowDimension = rDimension;
        this.columnDimension = cDimension;
        this.matrixData = mData;
    }

    public int getEntry(int rowPosition, int columnPosition) {
        return this.matrixData[rowPosition][columnPosition];
    }

    public int[] getRowVector(int i) { return this.matrixData[i].clone(); }

    // CORRECCIÓN: se usa rowDimension (no columnDimension) para el tamaño del vector columna
    public int[] getColumnVector(int i) {
        int[] columnVector = new int[this.rowDimension];
        for (int k = 0; k < this.rowDimension; k++) {
            columnVector[k] = matrixData[k][i];
        }
        return columnVector;
    }

    public int getRowDimension()    { return this.rowDimension; }
    public int getColumnDimension() { return this.columnDimension; }

    @Override
    public String toString() {
        StringBuilder returnValue = new StringBuilder();
        for (int i = 0; i < this.rowDimension; i++) {
            returnValue.append(Arrays.toString(this.matrixData[i])).append("\n");
        }
        return returnValue.toString();
    }

    // Imprime una matriz double[][] con formato legible
    public static void printDoubleMatrix(double[][] m) {
        for (double[] row : m) {
            System.out.println("   " + Arrays.toString(row));
        }
    }

    public abstract double[][] multiply(Matrix secondMatrix);
    public abstract double[][] invert();
}

// ============================================================================
// IMPLEMENTACIÓN SECUENCIAL | algoritmos sin paralelismo
// ============================================================================
class SequentialMatrix extends Matrix {

    public SequentialMatrix(int rDimension, int cDimension) {
        super(rDimension, cDimension);
    }

    public SequentialMatrix(int rDimension, int cDimension, int[][] mData) {
        super(rDimension, cDimension, mData);
    }

    // Multiplicación tradicional O(N³): fila de A × columna de B
    @Override
    public double[][] multiply(Matrix secondMatrix) {
        if (this.columnDimension != secondMatrix.getRowDimension()) {
            throw new IllegalArgumentException("Dimensiones incompatibles para multiplicación.");
        }
        int m = this.rowDimension;
        int n = this.columnDimension;
        int p = secondMatrix.getColumnDimension();
        double[][] resultMatrix = new double[m][p];

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < p; j++) {
                double sum = 0;
                for (int k = 0; k < n; k++) {
                    sum += this.matrixData[i][k] * secondMatrix.getEntry(k, j);
                }
                resultMatrix[i][j] = sum;
            }
        }
        return resultMatrix;
    }

    // Inversión por Eliminación de Gauss-Jordan O(N³)
    // Construye [A|I], reduce a [I|A⁻¹] mediante operaciones de fila
    @Override
    public double[][] invert() {
        if (this.rowDimension != this.columnDimension) {
            throw new IllegalArgumentException("La matriz debe ser cuadrada para invertirse.");
        }
        int n = this.rowDimension;

        // Construir matriz aumentada [A | I]
        double[][] augmented = new double[n][2 * n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                augmented[i][j] = this.matrixData[i][j];
            }
            augmented[i][i + n] = 1.0; // Identidad en la parte derecha
        }

        for (int i = 0; i < n; i++) {
            // Pivoteo parcial: buscar la fila con el mayor valor en la columna i
            int pivotRow = i;
            for (int k = i + 1; k < n; k++) {
                if (Math.abs(augmented[k][i]) > Math.abs(augmented[pivotRow][i])) {
                    pivotRow = k;
                }
            }
            // Intercambiar filas si el pivote no está en la posición actual
            if (pivotRow != i) {
                double[] temp = augmented[i];
                augmented[i] = augmented[pivotRow];
                augmented[pivotRow] = temp;
            }
            // Si el pivote es ~0, la matriz es singular (no invertible)
            if (Math.abs(augmented[i][i]) < 1e-9) {
                throw new ArithmeticException("Matriz singular: no tiene inversa.");
            }

            // Normalizar la fila pivote dividiendo por el valor del pivote
            double pivotValue = augmented[i][i];
            for (int j = i; j < 2 * n; j++) {
                augmented[i][j] /= pivotValue;
            }

            // Eliminar el elemento en la columna i para todas las demás filas
            for (int k = 0; k < n; k++) {
                if (k != i) {
                    double factor = augmented[k][i];
                    for (int j = i; j < 2 * n; j++) {
                        augmented[k][j] -= factor * augmented[i][j];
                    }
                }
            }
        }

        // Extraer la parte derecha de la matriz aumentada: esa es la inversa
        double[][] inverseMatrix = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                inverseMatrix[i][j] = augmented[i][j + n];
            }
        }
        return inverseMatrix;
    }
}

// ============================================================================
// CLASE PRINCIPAL: genera el dataset de tiempos en formato CSV
// ============================================================================
public class Main {

    public static void main(String[] args) {
        System.out.println("==================================================");
        System.out.println("DEMO: MATRICES DE PRUEBA Y RESULTADOS");
        System.out.println("==================================================");
        mostrarDemoMatrices();

        System.out.println("\n==================================================");
        System.out.println("DATASET DE TIEMPOS | MULTIPLICACIÓN E INVERSIÓN");
        System.out.println("==================================================");
        generarDatasetCSV();
    }

    // Muestra matrices de prueba con valores conocidos y sus resultados
    private static void mostrarDemoMatrices() {     

        // --- DEMO MULTIPLICACIÓN ---
        System.out.println("\n>>> MULTIPLICACIÓN DE MATRICES (2x2)");
        int[][] datosA = { {1, 2}, {3, 4} };
        int[][] datosB = { {2, 0}, {1, 2} };
        Matrix matA = new SequentialMatrix(2, 2, datosA);
        Matrix matB = new SequentialMatrix(2, 2, datosB);

        System.out.println("Matriz A:");
        System.out.print(matA);
        System.out.println("Matriz B:");
        System.out.print(matB);

        double[][] resMult = matA.multiply(matB);
        System.out.println("Resultado A x B:");
        Matrix.printDoubleMatrix(redondear(resMult));

        // --- DEMO INVERSIÓN ---
        System.out.println("\n>>> INVERSIÓN DE MATRIZ (2x2)");
        int[][] datosInv = { {4, 7}, {2, 6} };
        Matrix matInv = new SequentialMatrix(2, 2, datosInv);

        System.out.println("Matriz original:");
        System.out.print(matInv);

        double[][] resInv = matInv.invert();
        System.out.println("Inversa (Gauss-Jordan):");
        Matrix.printDoubleMatrix(redondear(resInv));

        // Verificación: A x A⁻¹ debe dar la identidad
        System.out.println("Verificación A x A-1 (debe ser identidad):");
        int n = 2;
        double[][] verificacion = new double[n][n];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                for (int k = 0; k < n; k++)
                    verificacion[i][j] += datosInv[i][k] * resInv[k][j];
        Matrix.printDoubleMatrix(redondear(verificacion));
    }

    // Mide tiempos de multiplicación e inversión para matrices de tamaño 40 a 400
    private static void generarDatasetCSV() {
        int inicio = 40;
        int fin    = 400;
        int salto  = 40;

        System.out.println("Dimension | Tiempo_Multiplicacion_ms | Tiempo_Inversion_ms");
        Random rand = new Random();

        for (int dim = inicio; dim <= fin; dim += salto) {
            // Matrices aleatorias para multiplicación
            Matrix m1 = new SequentialMatrix(dim, dim);
            Matrix m2 = new SequentialMatrix(dim, dim);

            // Matriz diagonal-dominante para garantizar que sea invertible
            int[][] datosSeguros = new int[dim][dim];
            for (int i = 0; i < dim; i++) {
                for (int j = 0; j < dim; j++) {
                    datosSeguros[i][j] = rand.nextInt(10);
                }
                datosSeguros[i][i] += 5000; // Diagonal grande asegura no-singularidad
            }
            Matrix mInvertible = new SequentialMatrix(dim, dim, datosSeguros);

            // Medir tiempo de multiplicación
            long tInicioMult = System.nanoTime();
            m1.multiply(m2);
            long tFinMult = System.nanoTime();
            double tMultMs = (tFinMult - tInicioMult) / 1_000_000.0;

            // Medir tiempo de inversión
            long tInicioInv = System.nanoTime();
            mInvertible.invert();
            long tFinInv = System.nanoTime();
            double tInvMs = (tFinInv - tInicioInv) / 1_000_000.0;

            System.out.printf(Locale.US, "%d | %.3f | %.3f\n", dim, tMultMs, tInvMs);
        }
    }

    // Redondea todos los valores de una matriz double[][] a 1 decimal
    private static double[][] redondear(double[][] m) {
        for (int i = 0; i < m.length; i++)
            for (int j = 0; j < m[i].length; j++)
                m[i][j] = Math.round(m[i][j] * 10.0) / 10.0;
        return m;
    }
}
