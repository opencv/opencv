package org.opencv.test;

import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;

/**
 * Indicates a test which is known to fail.
 *
 * <p>
 * This would cause a failure to be considered as success and a success to be considered as failure,
 * with the benefit of updating the related issue when it has been resolved even unintentionally.
 * </p>
 *
 * <p>
 * To have an effect, the test class must be annotated with:
 *
 * <code>
 * &#064;RunWith(OpenCVJUnitRunner.class)
 * </code>
 * </p>
 */
@Retention(RetentionPolicy.RUNTIME)
@Target(ElementType.METHOD)
public @interface NotYetImplemented {
    String value() default "";
}
