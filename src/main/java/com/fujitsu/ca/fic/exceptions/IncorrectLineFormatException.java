package com.fujitsu.ca.fic.exceptions;

import java.io.IOException;

public class IncorrectLineFormatException extends IOException {
    private static final long serialVersionUID = 1L;

    public IncorrectLineFormatException(String message) {
        super(message);
    }

}
