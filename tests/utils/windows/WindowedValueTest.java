package utils.windows;

import org.junit.jupiter.api.Test;
import utils.windows.WindowedValue;

import static org.junit.jupiter.api.Assertions.*;

class WindowedValueTest {

    @Test
    void add() {
        WindowedValue win = new WindowedValue(3);
        assertEquals(0, win.getWindowLength());

        win.add(1.0);
        assertEquals(1, win.getWindowLength());

        win.add(2.0);
        win.add(3.0);
        assertEquals(3, win.getWindowLength());

        win.add(4.0);
        assertEquals(3, win.getWindowLength());
    }

    @Test
    void getSumAndAverage() {
        WindowedValue win = new WindowedValue(3);
        assertEquals(0, win.getSum());
        assertEquals(0, win.getAverage());

        win.add(5.0);
        assertEquals(5.0, win.getSum());
        assertEquals(5.0, win.getAverage());

        win.add(7.0);
        win.add(9.0);
        assertEquals(21.0, win.getSum());
        assertEquals(7.0, win.getAverage());

        win.add(2.0);
        assertEquals(18.0, win.getSum());
        assertEquals(6.0, win.getAverage());
    }

    @Test
    void getStd() {
        WindowedValue win = new WindowedValue(4);

        win.add(1.0);
        assertEquals(0.0, win.getStd());

        win.add(1.0);
        win.add(1.0);
        assertEquals(0.0, win.getStd());

        win.add(0);
        win.add(0);
        assertEquals(0.5, win.getAverage());
        assertEquals(0.5, win.getStd(), 0.01);

        win.add(2);
        win.add(2);
        assertEquals(1.0, win.getAverage());
        assertEquals(1.0, win.getStd(), 0.1);
    }
}