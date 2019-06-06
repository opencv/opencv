package org.opencv.test;

import org.junit.runners.BlockJUnit4ClassRunner;
import org.junit.runners.model.FrameworkMethod;
import org.junit.runners.model.InitializationError;
import org.junit.runners.model.Statement;

public class OpenCVJUnitRunner extends BlockJUnit4ClassRunner {

    public OpenCVJUnitRunner(Class<?> klass) throws InitializationError {
        super(klass);
    }

    @Override
    protected Statement methodBlock(FrameworkMethod method) {
        Statement statement = super.methodBlock(method);
        if (method.getAnnotation(NotYetImplemented.class) != null) {
            statement = new NotYetImplementedStatement(method, statement);
        }
        return statement;
    }
}
