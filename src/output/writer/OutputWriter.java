package output.writer;

import java.io.BufferedWriter;
import java.io.IOException;

public abstract class OutputWriter {
    public abstract void write(BufferedWriter writer) throws IOException;
}
