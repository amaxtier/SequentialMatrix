import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;
import static org.junit.jupiter.api.Assertions.*;

// ============================================================================
// PRUEBAS DE UNIDAD — JUnit 5
// Se validan multiplicación, inversión y manejo de casos borde
// ============================================================================
public class SequentialMatrixTest {

    private static final double DELTA = 1e-6; // Tolerancia para comparar doubles

    // --- PRUEBAS DE MULTIPLICACIÓN ---

    @Test
    @DisplayName("Multiplicación 2x2 con resultado conocido")
    void testMultiplicacion2x2() {
        int[][] datosA = { {1, 2}, {3, 4} };
        int[][] datosB = { {2, 0}, {1, 2} };

        Matrix matA = new SequentialMatrix(2, 2, datosA);
        Matrix matB = new SequentialMatrix(2, 2, datosB);
        double[][] resultado = matA.multiply(matB);

        System.out.println("=== Multiplicación 2x2 ===");
        System.out.println("Matriz A:");
        System.out.println(matA);
        System.out.println("Matriz B:");
        System.out.println(matB);
        System.out.println("Resultado A × B:");
        Matrix.printDoubleMatrix(resultado);

        // Resultado esperado: {{4,4},{10,8}}
        assertEquals(4.0,  resultado[0][0], DELTA);
        assertEquals(4.0,  resultado[0][1], DELTA);
        assertEquals(10.0, resultado[1][0], DELTA);
        assertEquals(8.0,  resultado[1][1], DELTA);
    }

    @Test
    @DisplayName("Multiplicación por matriz identidad devuelve la misma matriz")
    void testMultiplicacionPorIdentidad() {
        int[][] datos    = { {3, 5}, {2, 7} };
        int[][] identidad = { {1, 0}, {0, 1} };

        Matrix m  = new SequentialMatrix(2, 2, datos);
        Matrix id = new SequentialMatrix(2, 2, identidad);
        double[][] resultado = m.multiply(id);

        System.out.println("=== Multiplicación por identidad ===");
        System.out.println("Matriz original:");
        System.out.println(m);
        System.out.println("Resultado M × I:");
        Matrix.printDoubleMatrix(resultado);

        // M × I debe dar M
        assertEquals(3.0, resultado[0][0], DELTA);
        assertEquals(5.0, resultado[0][1], DELTA);
        assertEquals(2.0, resultado[1][0], DELTA);
        assertEquals(7.0, resultado[1][1], DELTA);
    }

    @Test
    @DisplayName("Multiplicación lanza excepción con dimensiones incompatibles")
    void testMultiplicacionDimensionesIncompatibles() {
        Matrix m2x3 = new SequentialMatrix(2, 3, new int[][]{ {1,2,3},{4,5,6} });
        Matrix m2x2 = new SequentialMatrix(2, 2, new int[][]{ {1,0},{0,1} });

        System.out.println("=== Dimensiones incompatibles (esperado: excepción) ===");
        System.out.println("Intento: M(2×3) × M(2×2)");

        // Una matriz 2×3 no puede multiplicarse por una 2×2 (columnas ≠ filas)
        assertThrows(IllegalArgumentException.class, () -> m2x3.multiply(m2x2));
        System.out.println("Excepción lanzada correctamente.");
    }

    // --- PRUEBAS DE INVERSIÓN ---

    @Test
    @DisplayName("Inversión Gauss-Jordan 2x2 con resultado conocido")
    void testInversion2x2() {
        int[][] datos = { {4, 7}, {2, 6} };
        Matrix mat = new SequentialMatrix(2, 2, datos);
        double[][] resultado = mat.invert();

        System.out.println("=== Inversión Gauss-Jordan 2x2 ===");
        System.out.println("Matriz original:");
        System.out.println(mat);
        System.out.println("Inversa obtenida:");
        Matrix.printDoubleMatrix(resultado);

        // Inversa esperada de {{4,7},{2,6}}: det=10, inversa={{0.6,-0.7},{-0.2,0.4}}
        assertEquals( 0.6,  resultado[0][0], DELTA);
        assertEquals(-0.7,  resultado[0][1], DELTA);
        assertEquals(-0.2,  resultado[1][0], DELTA);
        assertEquals( 0.4,  resultado[1][1], DELTA);
    }

    @Test
    @DisplayName("A × A⁻¹ debe ser aproximadamente la identidad")
    void testMultiplicarPorInversa() {
        int[][] datos = { {4, 7}, {2, 6} };
        Matrix mat = new SequentialMatrix(2, 2, datos);
        double[][] inversa = mat.invert();

        System.out.println("=== Verificación A × A⁻¹ ≈ I ===");
        System.out.println("Matriz A:");
        System.out.println(mat);
        System.out.println("A⁻¹:");
        Matrix.printDoubleMatrix(inversa);

        // Convertir la inversa a int[][] para poder usar multiply()
        // Se multiplica manualmente A × A⁻¹ para verificar identidad
        int n = 2;
        double[][] producto = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                for (int k = 0; k < n; k++) {
                    producto[i][j] += datos[i][k] * inversa[k][j];
                }
            }
        }

        System.out.println("A × A⁻¹ (debe ser ≈ I):");
        Matrix.printDoubleMatrix(producto);

        // Verificar que la diagonal es ~1 y el resto ~0
        assertEquals(1.0, producto[0][0], 1e-5);
        assertEquals(0.0, producto[0][1], 1e-5);
        assertEquals(0.0, producto[1][0], 1e-5);
        assertEquals(1.0, producto[1][1], 1e-5);
    }

    @Test
    @DisplayName("Inversión lanza excepción en matriz no cuadrada")
    void testInversionNoEsCuadrada() {
        Matrix mRect = new SequentialMatrix(2, 3, new int[][]{ {1,2,3},{4,5,6} });

        System.out.println("=== Inversión de matriz no cuadrada (esperado: excepción) ===");
        assertThrows(IllegalArgumentException.class, mRect::invert);
        System.out.println("Excepción lanzada correctamente.");
    }

    @Test
    @DisplayName("Inversión lanza excepción en matriz singular")
    void testInversionMatrizSingular() {
        // Filas linealmente dependientes → det = 0 → no invertible
        int[][] singular = { {1, 2}, {2, 4} };
        Matrix mat = new SequentialMatrix(2, 2, singular);

        System.out.println("=== Matriz singular (esperado: excepción) ===");
        System.out.println("Matriz:");
        System.out.println(mat);
        assertThrows(ArithmeticException.class, mat::invert);
        System.out.println("Excepción lanzada correctamente.");
    }

    public static void main(String[] args) {
        SequentialMatrixTest t = new SequentialMatrixTest();
        
        System.out.println("\n=== TEST 1 ===");
        t.testMultiplicacion2x2();
        
        System.out.println("\n=== TEST 2 ===");
        t.testMultiplicacionPorIdentidad();
        
        System.out.println("\n=== TEST 3 ===");
        t.testMultiplicacionDimensionesIncompatibles();
        
        System.out.println("\n=== TEST 4 ===");
        t.testInversion2x2();
        
        System.out.println("\n=== TEST 5 ===");
        t.testMultiplicarPorInversa();
        
        System.out.println("\n=== TEST 6 ===");
        t.testInversionNoEsCuadrada();
        
        System.out.println("\n=== TEST 7 ===");
        t.testInversionMatrizSingular();
        
        System.out.println("\nTodas las pruebas completadas.");
    }
}
