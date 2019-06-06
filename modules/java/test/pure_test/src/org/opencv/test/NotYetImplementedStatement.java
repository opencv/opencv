package org.opencv.test;

import org.junit.internal.AssumptionViolatedException;
import org.junit.runners.model.FrameworkMethod;
import org.junit.runners.model.Statement;

public class NotYetImplementedStatement extends Statement {

    private final FrameworkMethod method;
    private final Statement next;

    public NotYetImplementedStatement(FrameworkMethod method, Statement next) {
        this.method = method;
        this.next = next;;
    }

    @Override
    public void evaluate() throws Exception {
        boolean complete = false;
        try {
            next.evaluate();
            complete = true;
        } catch (AssumptionViolatedException e) {
            throw e;
        } catch (Throwable e) {
            // expected
        }
        if (complete) {
            throw new AssertionError("Method " + method.getName() + "() is marked as @NotYetImplemented, but already works!");
        }
    }
}
